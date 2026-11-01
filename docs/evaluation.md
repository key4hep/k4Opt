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

- `stage = "simulation"`: the framework makes that stage of each sample the evaluator measures
  available as the problem's `samples` say (runs it, or takes its file) and passes its file as
  `request.inputs["simulation"]`, along with the files of the stages before it; the evaluator is
  called once per sample, each time with one sample's files, events and seed, and never builds a
  command, starts a tool or handles its failures; a stage that several evaluators read runs once
  (see [Stages](#stages)). The evaluator's entry in the problem states the same `stage`, so the
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
  without `check`, and so does the check of the settings against what the evaluator declares
  (`check_settings`, which `Evaluator.check` calls): when a session opens, before any
  candidate, since they need none. `check` itself runs once per candidate, with a request made
  for that candidate, before materialization, and that request is reused during its execution.
  `check` is for the settings and the contract: what it raises stops the evaluation before
  anything is created, as a misuse. A candidate the evaluator finds scientifically unusable is not refused there:
  `evaluate()` says so with `self.invalid("...")`, and the candidate is recorded as `invalid`.
- `metrics`: lets `k4opt evaluate` reject an unsupported objective before anything runs, and
  holds every returned measurement of those metrics to the declared unit, so one metric identity
  never carries two units. An objective names the binding that measures it, so two evaluators
  may declare metrics of the same name, and an evaluator that declares none is checked when its
  result arrives. A metric of another stage is written `("analysis", "rms")`;
  declaring one metric twice (`"rms"` and `("simulation", "rms")`) is an error, and so is a unit
  that is not a nonblank string (`"1"` for a dimensionless metric). Both are checked before
  anything runs.
- `artifacts`: the files a `valid` result returns besides its measurements, each name with what
  the file holds (`{"histogram": "shower widths, JSON {edges, counts}"}`). A `valid` result must
  return every one, and no result another; an evaluator that declares none may return any (see
  [Artifacts and the files of a run](#artifacts-and-the-files-of-a-run)).
- the return value: a mapping of metric to a number or to per-event numbers (a list, a numpy
  array, ...; `Measurement.from_values`) becomes a `valid` result, with the unit from `metrics`.
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

## Writing a combined evaluator

A combined evaluator measures a candidate using results from several samples. It declares
`roles` instead of a `stage`; the problem wires each role to a per-sample binding or a stage.
For example, this evaluator reads histograms produced by `calo.shower_shape`:

```python
from k4opt.evaluation import Evaluator, Role


class Separation(Evaluator):
    name, version = "calo.separation", "1"
    roles = {
        "signal": Role(artifacts=("histogram",)),
        "background": Role(artifacts=("histogram",)),
    }
    settings = {"signal_efficiency": float}
    metrics = {("analysis", "background_rejection"): "1"}

    def evaluate(self, request):
        (signal,) = request.roles["signal"]
        (background,) = request.roles["background"]
        # Read signal.artifacts["histogram"].path and background's histogram,
        # calculate rejection, then return an EvaluationResult or metric mapping.
```

The complete [Separation example](../examples/evaluators/separation.py) implements this
calculation; [the runnable problem](../examples/problem/cld-ecal-pi0-rejection.yaml) supplies
photon and pi0 gun samples. See [its definitions and limits](evaluators.md#photonpi0-example).

`Role(metrics=(("analysis", "yield"),), artifacts=("histogram",))` requires each input's
producer to declare that metric and artifact. A valid upstream evaluation must return them.
`Role(many=True, ...)` accepts one or more inputs; otherwise a role takes exactly one. Either
way, `request.roles[name]` is a tuple of `RoleInput`. Role names are Python identifiers, not
keywords or names starting with `_`. Every declared role must be wired, with no extras.

Each `RoleInput` exposes:

- `sample`, `binding` (or `stage` for a stage input), and `label` (`sample/binding`);
- `measurements`, with `measurement(stage, metric)` for lookup;
- `artifacts`, named `Artifact` objects with an absolute `path`, logical `ref`, byte count and
  SHA-256 when hashed;
- `events`, `seed`, the sample's opaque `properties`, the producer's resolved `settings` and
  its result `provenance`.

A `Role(stage="simulation")` takes a stage file directly. Its input has
`artifacts["simulation"]`, with reference `sample/simulation#events`; generated EDM4hep files
are not hashed. Files supplied by the caller or named in the problem carry the input hash.
There are no collection helpers for stage roles yet; read these files with podio yourself.

Session opening checks wiring, cardinality, settings and declarations before creating files.
`check(request)` receives candidate values and settings, with **empty roles**: upstream data
does not exist yet. `evaluate(request)` runs only after every input succeeds; failed, invalid
or blocked inputs block it. Independent evaluations still run. One upstream task may feed
several roles or combined evaluators, and it executes only once. A fresh evaluator instance
is created for every invocation and candidate. Candidate-level evaluators cannot be role
inputs yet, so combined evaluators cannot be chained.

Records store each role's ordered input labels, and store upstream measurements and artifacts
once at their producer. `Assessment.from_summary` rechecks the wiring, required metrics and
blocking against the problem and recorded declarations. Studies compare role declarations,
wiring and sample properties when resuming.

Normalization and statistics belong entirely to evaluators. The test-only
[`ToySignificance`](../tests/support/toy_significance.py) demonstrates a single-bin Asimov
significance using cross sections from sample properties and generated/selected sums of
weights. It converts luminosity from ab⁻¹ to fb⁻¹ itself. This toy has no systematics, shape
information or correlations, and rejects nonpositive generated weight sums and background.
The framework preserves signed measurements and covariance artifacts without interpreting
them. A real sensitivity analysis needs suitable event files, reconstruction and a statistical
model; the toy is a data-flow test.

## Request

`EvaluationRequest` is immutable and every field is optional:

| Field | Meaning |
| --- | --- |
| `candidate` | parameter values for this evaluation, keyed by the canonical parameter identity (a geometry parameter: its canonical target such as `constants.ECal_cell_size`); booleans, integers, finite floats or strings. Not checked against bounds |
| `settings` | the evaluator-defined `settings` of its entry in `evaluators`, with the declared defaults and each quantity as a float in its declared unit; nested data, copied and read-only |
| `collections` | role to the name of the collection it reads: the one the problem's entry names, or the declared default |
| `inputs` | role name to local `Path` of an already prepared artifact: the main output of each stage it reads, by stage (`inputs["simulation"]`, an EDM4hep file), and the compact XML snapshot as `inputs["geometry"]`. The evaluator documents the roles it accepts |
| `workdir` | where the evaluator may write, and where its artifacts must be. `k4opt evaluate` gives each evaluation its own, `<output>/samples/<sample>/evaluators/<binding>` (`<output>/evaluators/<binding>` for one of the candidate), created for the call; the request never creates or checks it |
| `properties` | immutable JSON mapping from the measured sample, or empty for a candidate-level evaluator; k4opt records it without interpreting it |
| `roles` | immutable mapping of role name to tuples of `RoleInput`; empty during `check()` and for evaluators without roles |
| `events`, `seed` | those of the stage the evaluator reads, where that stage has them (a ddsim simulation); `None` otherwise. `events` is positive |

The request holds no `ProblemSpec`, geometry or stage object; the framework builds it from
those.

## Result

`EvaluationResult` is what `evaluate` returns:

| Field | Meaning |
| --- | --- |
| `status` | a `Status`, a `StrEnum` whose values are `"valid"`, `"invalid"`, `"failed"` and `"blocked"`. `valid`: ran, result scientifically usable. `invalid`: ran, but the result is not usable (the evaluator's own checks failed). `failed`: the execution failed. `blocked` is k4opt's alone, for an evaluator it did not call because something it reads did not succeed; an evaluator that returns it breaks the contract |
| `measurements` | tuple of `Measurement`, mixing stages if needed. Independent of the status: an `invalid` result may keep diagnostic values (they are never objectives), and a `valid` one may have none if it only produced artifacts; a problem that asks for a metric still requires it from a `valid` result |
| `artifacts` | name to local output `Path`, possibly empty. A name starts with a letter or digit and holds only letters, digits, `.`, `_` and `-`, since it is part of the artifact's reference (`gamma_10gev/calo.photon#histogram`). Existence is not checked at construction; k4opt checks each file and hashes it when the evaluation returns |
| `message` | explanation; required for `invalid`, `failed` and `blocked` |
| `provenance` | optional JSON-compatible metadata the evaluator chose to supply |

A `Measurement` is `stage`, `metric`, a finite `value` (booleans, NaN and infinity are
rejected), an optional descriptive `unit` (never converted) and an optional symmetric,
nonnegative `uncertainty`. A numpy number is stored as the Python `int` or `float` it stands
for, here and in `provenance`. An omitted uncertainty means *not provided*, not zero. Each
`(stage, metric)` occurs once per result. A missing measurement is absent, never `0`, `inf` or
a fake success.

`Measurement.from_values(stage, metric, values, unit)` averages per-event values: the value
is their unweighted mean and the uncertainty the standard error of the mean, `s / sqrt(N)` with
`s` the sample standard deviation (`ddof=1`); a single value has no uncertainty. Every value
counts once, so events with weights (signed ones included) need the evaluator's own weighted
estimate. Statistical uncertainty only: it says nothing about systematic effects.

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

### Artifacts and the files of a run

An evaluator writes its files in `request.workdir` and returns those it wants recorded as
`artifacts`:

```python
class ShowerShape(Evaluator):
    name, version, stage = "calo.shower_shape", "1", "simulation"
    collections = {"hits": Collection("SimCalorimeterHit")}
    metrics = {"ecal_barrel.shower_width": "mm"}
    artifacts = {"histogram": "shower width per event, JSON {edges, counts}"}

    def evaluate(self, request):
        widths = [...]
        path = request.workdir / "histogram.json"
        path.write_text(json.dumps({"edges": edges, "counts": counts}))
        return EvaluationResult.valid(
            [Measurement.from_values("simulation", "ecal_barrel.shower_width", widths, "mm")],
            artifacts={"histogram": path},
        )
```

When the evaluation returns, k4opt checks each artifact, whatever the status: it is a file in
the evaluation's own directory (a link is followed, so one that leads out of the directory is
refused), it is declared if the evaluator declares `artifacts`, and a `valid` result returns
every declared one. A violation is an `EvaluationContractError`. Each artifact is then hashed,
once, and recorded as an `Artifact`: its reference `<task>#<name>`
(`gamma_10gev/calo.shower_shape#histogram`), which stays the same wherever the run directory is
moved, its path, its size and its SHA-256.

The output of a stage that ran is recorded the same way (`gamma_10gev/simulation#events`), but is
never hashed: an EDM4hep file can be gigabytes, so its `sha256` is `null`. A record claims a
content identity only where it computed one; nothing can later prove that the bytes of an
unhashed output are unchanged. While the candidate runs, each file a task made is watched by its
size, times and inode, kept in memory and never recorded: a task whose input file changed after
it was made fails without running, one whose input changes while it runs fails with what it
returned kept as diagnostics, and if any file a task made changed by the end of the candidate,
no `summary.json` is written. The files of the candidate's geometry are watched the same way,
as a whole: every simulation and every evaluation reads the geometry, so a file of the snapshot
that changes, goes or is added fails the tasks that read it after the change, and leaves no record
(the fingerprint describes the snapshot as materialized; the samples simulated after an edit would
not have the geometry it describes). A hashed file whose stat changed is hashed again, and only
other bytes count as a change. A file rewritten with the same size within one tick of the kernel's
clock (milliseconds) keeps its stat, so it is not told apart.

Every path a record holds of a file in its run directory is relative to that directory
(`samples/gamma_10gev/simulation/events.edm4hep.root`); the directory is recorded once, as
`run_dir`, where it was made. A record, or a study of them, therefore stays true when it is moved
or archived: its files are found relative to the `summary.json` that names them. Files outside
the run directory (a generation file of the problem, a supplied one, the geometry's source) keep
their absolute path with their SHA-256, and the `argv` of a stage is the command exactly as it
ran.

### Reading EDM4hep files

`self.events(request)` is `read_events` on the file of the stage in `stage`, with the declared
`collections` under the names the request gives them. `read_events(path, {role: (collection,
value_type)})` yields an `Event` per event (podio frame) of an EDM4hep file, in file order, whose
attributes are the roles: each the list of that collection's elements in this event. All the
collections of an `Event` come from the same frame, so they belong to the same physics event.
Nothing is matched up across files: `request.read(stage, collection, value_type)` reads one
collection of any stage's file, and two collections read that way are not known to be aligned
event by event. `read_collection(path, collection, value_type)` is the one-collection form,
`available_collections(path)` lists the collections of a file without reading any,
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

The stages of the pipeline are made available by the framework, not by an evaluator. Each sample
of the problem says how its stages are made: each is run by an `executor` or given as a `file`
(see [problem](problem.md#samples)). A stage turns the stages before it (and the candidate's
geometry) into a file, and an evaluator whose `stage` names it reads that file from
`request.inputs`. Each stage of a sample runs at most once per candidate, however many evaluators
read it, so several evaluators share one expensive simulation, and an evaluator can be tested
with a small fixture file and no Key4hep stack. The samples share nothing but the candidate's
geometry: each is simulated on its own, and an evaluator that measures two samples is called
twice, once with each sample's files.

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
(logs, the command) are its logs.

A stage asks only for what it uses: `simulation` needs `inputs["geometry"]` and a `workdir`,
`generation` neither, and an evaluator that reads none runs no stage at all.

The work of a candidate is a set of **tasks**, each run at most once: the candidate's
`geometry`, every stage of each sample that its evaluators read (`gamma_10gev/simulation`), and
every **evaluation**: an evaluator on one of its samples (`gamma_10gev/calo.photon`), or on the
candidate (`readout`), for one that reads no stage. Each task names the tasks it reads, and
nothing else orders it: a stage that runs reads the stages of its sample before it (and the
geometry, if it is made from it). Every evaluation reads the geometry. A per-sample evaluation
also reads its stage and that stage's inputs; a combined evaluation reads the producers named
by its roles.
Each writes in a directory of its own (`samples/<sample>/simulation/`,
`samples/<sample>/evaluators/<binding>/`, `evaluators/<binding>/` for one of the candidate). A
task runs only when every task it reads succeeded (a stage is `done`, an evaluation `valid`);
otherwise it is **`blocked`** and not run, so the order in which the tasks run never changes a
result. A stage that fails (a file it cannot read, the tool missing, a nonzero exit, no output,
an output without the requested events) is `failed` with its message; every task that reads it
is `blocked`, an evaluation too, without being called ("not run: gamma_10gev/simulation
failed"). A task is blocked by the tasks that failed, not by the blocked ones between them.
Tasks that do not depend on it, the other samples' among them, still run and keep their results.

An `inputs[stage]` supplied by the caller for a stage of a sample (`k4opt evaluate --input
gamma_10gev/simulation=FILE`) replaces running that stage and the ones of the sample only it
needs, and it is authoritative: a supplied `generation`
file is simulated even if the problem describes a particle gun. The record identifies it (the
path, size and SHA-256 under `provided`; a file that could not be hashed carries only its path,
so a verified input is distinguishable from one identified by its name) and lists it as
`supplied`. A supplied stage that the parameters change (a simulation, when they vary the
geometry) was not made from the candidate: if a required evaluation reads it, the candidate
record says so, `represents_candidate = False`, and must never count as an evaluation of that
candidate, whatever its status. One that only diagnostics read leaves the record
representative. A file the problem gives (`file:`) or the caller supplies is hashed once
per file version, not once per candidate: a file is told apart by size, mtime, ctime and inode,
so one rewritten within a few milliseconds of being hashed is not, and files are assumed stable
while a campaign runs.

The stages are recorded once per candidate, under their sample, apart from the evaluators'
results: what ran (the
command, exit code and time limit), what was given or supplied, the files a ddsim option names (with their
hashes), the events found, its status (`done`, `failed`, `blocked`) and message, the tasks it
reads and those that blocked it, its `directory`, its `outputs` (the main output of a stage that
ran and is `done`, [unhashed](#artifacts-and-the-files-of-a-run)) and `logs`, and the time spent
on it. An
evaluator's result holds only what it returned, so its artifact and provenance names cannot
clash with the stages'. The stages take their conditions from their sample, never from an
evaluator, except a time limit, which is the caller's: a stage that runs a program (ddsim)
longer than the `timeout` of the session kills it and fails. An evaluator has no time limit: one
that never returns holds its job, or with one job the session, until the session is stopped
(Ctrl-C, SIGTERM). Stages are internal for now
(`k4opt.problem._stages`).

## Loading an evaluator

`load_evaluator("path/to/file.py:Class")` (or `"package.module:Class"`) imports the file or
module and returns the class, after checking that it is a class with an `evaluate` method;
`EvaluatorLoadError` otherwise. It creates no instance: k4opt creates one for every candidate
([Lifecycle](#lifecycle-a-class-and-an-instance-per-candidate)). This is what
`k4opt evaluate --evaluator` uses. The file is executed as ordinary Python: name only evaluators you trust. Like `python file.py`, its
directory is put on `sys.path` (and stays there), so an evaluator can import helper modules next
to it, at the top of the file or inside a method. Python caches modules by name for the whole
process, so two evaluator files that each import a helper called `metrics` share whichever was
imported first. For one study this is fine. When several studies run in one process, give the
helpers unique names or, better, install the study as a package and use
`package.module:Class`; the same holds for the several evaluators of one problem. It is not a
registry; the `name` of each entry of `evaluators` stays a label that must equal the `name` of
an evaluator that is loaded.

`evaluator_source(cls)` reports the path, size and SHA-256 of the file that defines an
evaluator class, taken when `load_evaluator` imports it, and whether those are the bytes Python
`executed`. For `FILE.py:Class` they are (`executed: true`): the bytes hashed are the bytes
compiled. For `package.module:Class` the file is read right after the import
(`executed: false`); Python may have run a cached `.pyc` of it, which only an edit that keeps
size and timestamp leaves stale. The class keeps that identity if its file is later edited or
removed. A class whose module was imported before `load_evaluator`, or reloaded since
(`importlib.reload`), carries only its path: hashing the current file would not establish which
code Python executed. `reference` is what loads the class again: `FILE.py:Class` with the
file's absolute path, or `package.module:Class` as given; `null` for a class `load_evaluator`
did not load. It is provenance only: a study compares the SHA-256, not where the file is. Only the defining file is
identified, not imported helpers. To label those, `revision` holds the commit and dirty flag of
the git checkout that tracks the file, taken at the same time: an edited helper module of that
checkout shows as `dirty: true`. It is `null` for a file that no checkout tracks (outside one,
or new or ignored in it), and helpers outside a checkout are not tracked at all.

## Lifecycle: a class, and an instance per candidate

k4opt is given evaluator classes, not instances: `EvaluationSession.open(problem, [CellCount])`,
`optimize(problem, [CellCount], ...)` and `k4opt evaluate --evaluator FILE.py:CellCount` all
take the class. A function that builds an instance, or an instance, is refused: it could set
up the evaluator in ways its class's source, which the records identify as the code that ran,
does not say. When a session opens, k4opt creates one instance of each to read what it
declares and bind it to the entries of `evaluators` that name it; that instance is asked
nothing else. Every candidate then gets a new instance for each binding, before anything of the
candidate is created: k4opt calls its `check()` and later its `evaluate()`, then drops it. Nothing an
evaluator keeps on `self` reaches another candidate.

Every instance must declare what the first did: the same class, `name`, `version`, `stage`,
`settings`, `collections`, `metrics`, `artifacts` and `roles`. The records name the first one's class as the code that
ran, its metrics decide what the objectives may name, and its settings and collections, defaults included,
make the request, which must not change from one candidate to the next. What the first declared
is copied all through when the session opens, so an evaluator that changes a nested default of
its class (`self.settings["cuts"]["min"] = ...`) is caught at the next candidate. An instance
that declares otherwise, or a constructor that raises, is an `EvaluationContractError` before
anything of the candidate is created. Setup that is expensive and the same for every candidate
(a model file, a lookup table) belongs in a module-level cache of the evaluator's own
(`functools.cache`), not in `__init__`, which runs for every candidate.

With more than one job ([execution](#execution-jobs-and-worker-processes)), `evaluate()` runs
in a worker process, which loads the evaluator again by its reference (`FILE.py:Class` or
`module:Class`): give classes [`load_evaluator`](#loading-an-evaluator) loaded, not ones
defined in the script that runs the session. The instance checked before anything of the
candidate is created then only does that: the worker creates the instance that evaluates, calls
its `check()` with the same request and then its `evaluate()`. So `check()` and `evaluate()`
always reach one instance, and `check()` runs twice; keep it free of side effects. What
`evaluate()` prints goes to `evaluator.log` in its own directory, which the summary names
(`log`); it writes only in `request.workdir` and shares nothing with the process that started
the session. A module-level cache lives as long as its worker, which serves many evaluations.

## Problem, evaluator, record

- **Problem** (`k4opt.problem`): the declared parameter domains and the requested objective and
  constraint metrics.
- **Evaluator**: user-written code that accepts an `EvaluationRequest` and returns an
  `EvaluationResult`; the documentation of its version defines what its metrics mean.
- **`EvaluationRecord`**: the request and result of **one evaluator's attempt** at one
  candidate, on one sample or on the candidate, plus what the framework observed. It holds only that evaluator's own work; the
  stages it read are recorded once for the candidate.

| `EvaluationRecord` field | Source |
| --- | --- |
| `request` | the object the evaluator received |
| `result` | what the evaluator returned, or the result that stands for it when it was not called: `blocked` when a stage it reads did not succeed, `failed` when its stage's file lacks a collection it declares |
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
candidate = _evaluate_candidate(spec, {"constants.ECal_cell_size": 6.0}, [CellCount])
```

In order, it:

1. validates the candidate (`CandidateError` otherwise): exactly one value per declared
   parameter, keyed by `parameter.geometry.canonical_target`; continuous `int`/finite `float`
   within inclusive bounds; integer a real `int` (no `bool`, no rounding) within bounds;
   categorical one of the choices by kind (strings need a string, booleans a boolean, numbers
   compare numerically). Units are never interpreted or converted;
2. matches the evaluator classes to `evaluators` by the `name` of an instance of each, in any
   order: each evaluator the problem names must be given exactly once, however many entries
   bind it, with the declared `version` and reading the declared `stage`, and no other
   (`EvaluationContractError` otherwise, and for a class that cannot be created). Each entry is
   a binding, identified by its binding name (`as`, its `name` when not given), and each
   evaluation by the binding and the sample it measures (`gamma_10gev/calo.photon`); tasks,
   records, run directories and the objectives and constraints use that identity;
3. before anything runs, checks the stages of each sample the evaluators read and that each
   objective and constraint names a metric its evaluation's evaluator declares, in the declared
   unit (see [problem](problem.md#evaluators); `EvaluationContractError` otherwise), then
   creates a new instance of the evaluator of each evaluation for the candidate, resolves its
   settings and collections and calls its `check()`. A collection ddsim is not expected to
   write, two bindings that measure the same twice on a sample and two samples written alike are
   warnings, kept with the candidate;
4. runs the candidate's [tasks](#stages), each after the tasks it reads, in this order: the
   geometry (`inputs["geometry"]`, already prepared), then for each sample every stage its
   evaluators read, once each, and its evaluations in problem order, then the evaluations of the
   candidate. A stage whose file `inputs` supplies (`inputs["gamma_10gev/simulation"]`) is not
   run, and a stage the problem gives as a `file` is read; another runs in `workdir`. The output of
   the `simulation` stage, run or supplied, must hold exactly its `events` events (at least one
   when the problem gives the file), otherwise the stage fails ("incomplete simulation"); the
   number found is recorded as `events_read`;
5. calls each evaluator's `evaluate()` once with a request built from the validated candidate,
   its own `settings` (defaults filled in, quantities converted) and `collections`, the
   `events` and `seed` of the stage it reads of its sample and the files of the tasks it reads,
   in its own `workdir/samples/<sample>/evaluators/<binding>` (`workdir/evaluators/<binding>` for
   an evaluation of the candidate). An evaluation whose stage did not succeed is `blocked` instead,
   and one whose stage's file lacks a collection it declares (see
   [Collections](#collections)), or reads a file that changed after its task made it, is
   `failed`; neither is called. `EvaluationFailed` and
   `EvaluationInvalid` become a `failed` or `invalid` result; other exceptions of an evaluator
   propagate;
6. checks each result; a violation is an `EvaluationContractError`, an inconsistent evaluator
   rather than a violated constraint. A `valid` result must report every metric the objectives
   and constraints name of its evaluation (extra metrics are fine) in the `unit` a constraint
   declares; and if the evaluator declares `metrics`, every
   measurement of one, whatever the status, must carry the declared unit. Each artifact must be
   a file in the evaluation's own directory and as the evaluator declares them; it is hashed then
   ([Artifacts](#artifacts-and-the-files-of-a-run)). Whether a value satisfies `min`/`max` is not
   judged here but by the session's
   [assessment](#assessment-objectives-limits-and-feasibility).

The `CandidateRecord` holds the plan of its tasks and the result of each: the geometry, the
stages of each sample (each recorded once; `stage(name, sample)`) and one `EvaluationRecord` per
evaluation in plan order (`evaluations`, by label, `gamma_10gev/calo.photon` or `readout`;
`record(label)`; `records`), and the time of the whole evaluation. Like the records it holds, it cannot be changed once made.

Each task of the plan is *required* or a *diagnostic*. The required tasks are the evaluations
whose metrics an objective or a constraint names, those of a binding marked `required: true`
([problem](problem.md#evaluators)), and every task these read, transitively: the stages of their
samples, the evaluations a required combined evaluator reads, the geometry. Its `status` is the
candidate's, decided by the required tasks alone:

| Required tasks | Candidate | Meaning |
| --- | --- | --- |
| every stage `done`, every evaluation `valid` | `valid` | complete and usable; every objective and constraint has a value |
| at least one `failed` | `failed` | technically incomplete: a stage or an evaluator did not run to the end |
| none `failed`, at least one `invalid` (and what it blocked) | `invalid` | at least one evaluator found the result unusable |

`failed` outranks `invalid`, so an incomplete evaluation is never reported as a scientific
verdict. A required task is only ever blocked by another required one, so one that is blocked
without a failure was blocked by an `invalid` result: the candidate is `invalid`, and what read
that result was not run. Its `message` names every required task that failed or is invalid,
with its status and message, then the required tasks they blocked
(`gamma_10gev/simulation failed: ddsim exited with status 2 ...; blocked:
gamma_10gev/calo.photon`), so a mixed outcome shows both. Every sample a required evaluation
measures counts: one that fails makes the candidate `failed`, whatever the others measured.

A diagnostic runs and is recorded like any other task, but changes neither the status nor the
message. `diagnostics` holds the result of each diagnostic that did not succeed, in plan order;
the summary lists them (`task`, `status`, `message`) beside the `required` task labels. `value(MetricRef(evaluator, stage, metric,
sample))` gives an objective or constraint metric only from the `valid` result of the evaluation
it names: a missing value is never made up, and the diagnostics of an `invalid` result
never stand in for it.
The measurements of the other evaluations are kept whatever happened to one of them.

Every second is counted once: a stage's time (running it and checking its output) is recorded
with the stage, the geometry's with the geometry, an `EvaluationRecord`'s `elapsed_seconds` is
its own `evaluate()` call, and the `CandidateRecord`'s is the whole evaluation: every task and
the checks.

Each task that ran also records its peak memory, in bytes, as `peak_memory_bytes` (`null` for one
that did not run):

- `process`: the peak resident memory of the k4opt process that did the task's work (a worker
  process, or the session's own with one job) while it did it, all it held included (Python,
  podio, ROOT, the evaluator). It is measured per task: the process's high-water mark is reset
  before each task (`/proc/self/clear_refs`, Linux), so a worker that is reused does not carry
  one task's peak into the next. A worker resets it before it creates the evaluator's instance,
  so the instance's creation and its `check()` count too; with one job they ran in the preflight,
  before the task, and do not. `null` where the system cannot reset it.
- `program`: for a stage that ran a program (ddsim), that program's peak, as the kernel kept it
  when it ended (`wait4`): the largest of ddsim's own and of each process it started and waited
  for (a wrapper script's child), each on its own, never their sum. `null` for every other task,
  and for a program that did not end by itself (killed at its time limit).

`peak_memory(summary)` reads them back, by task label, as `PeakMemory(process, program)`; its
`per_task` is both added: an **estimate** of what the task needs while it runs, however many
jobs it holds (a simulation's threads share its memory), not a bound. The
two peaks need not come at once, which makes it high; a program that runs several processes at
once is counted by the largest of them, which makes it low (ddsim is one process, its Geant4
threads share its memory). A true bound of a process tree needs the kernel's accounting of a
cgroup, which k4opt does not set up. `k4opt evaluate` prints the task that needs most
(`memory:      1.93 GiB per task (estimate), for gamma_10gev/simulation (k4opt 312 MiB, ddsim
1.62 GiB)`). Like the times, they say how the candidate ran, never what was evaluated: no
comparison of records reads them.

## Evaluating candidates: `EvaluationSession`

`k4opt.problem.EvaluationSession` evaluates candidates of one problem file,
each into a run directory of its own with a `summary.json`, exactly as `k4opt evaluate` does
for one ([evaluators](evaluators.md#k4opt-evaluate)). It knows no optimizer: a scan by hand, a
script and an optimization library call it the same way, with candidates given as values.

```python
from k4opt.evaluation import load_evaluator
from k4opt.problem import EvaluationSession

session = EvaluationSession.open(
    "examples/problem/cld-ecal-granularity.yaml",
    [load_evaluator("examples/evaluators/calo_photon.py:CaloPhoton")],
    timeout=3600,  # seconds ddsim may take per candidate; no limit without it
)
for size in (5.0, 7.0, 10.0):
    outcome = session.evaluate({"constants.ECal_cell_size": size}, f"runs/scan/{size}mm")
    print(size, outcome.candidate.status, outcome.assessment.feasible)
```

| Once, when the session opens | For every candidate, in `check()` and `evaluate()` |
| --- | --- |
| load the problem file (later edits to it do not reach the session) and discover the geometry | check the candidate against the parameter domains |
| bind the evaluator classes through one instance of each, check the stages of each sample they read and their settings and collections, decide who measures each metric | create a new instance of the evaluator of each evaluation, which must declare what the first did, build its request for this candidate and call its `check()` |
| collect the warnings about collections ddsim is not expected to write, and about what is measured or simulated twice (`session.warnings`); list the diagnostics (`session.diagnostics`, and `session.notice` to tell whoever starts it) | check that every file the stages read, what the geometry's external references read, `KEY4HEP_STACK`, `K4GEO` and k4opt's code are still what they were when the session opened |
| record the size and SHA-256 of every file the stages read, by where the problem names it (a stage `file`, `samples.gamma_10gev.generation.file`; a file a ddsim option names; a file supplied with `inputs`, `supplied.gamma_10gev.simulation`; a file two samples read is hashed once and named for each), what the geometry's external references read ([`external_files`](geometry.md#fingerprint)), and each evaluator's source identity and declared metrics | `evaluate()` only: materialize the geometry, run the stages, call the evaluators, check again what the previous row checks, assess, write the summary |

No request or evaluator instance outlives the candidate it was made for, so an evaluator's
`check()` and `evaluate()` always see that candidate's values, and nothing an instance keeps
reaches another candidate. `check(candidate)` creates and runs nothing: it raises for
what `evaluate()` would refuse before creating anything. A file the stages read (named by the
problem or supplied) that changes during a session (other bytes under the same path, or removed)
stops the next candidate with a `RunError` before anything is created; `verify_inputs()` checks
it on its own. So does a change of what the geometry's external references read, of
`KEY4HEP_STACK` or `K4GEO`, or of k4opt's files: the candidates of a session run in the
environment it opened in, which every summary of the session records. A change while a candidate
runs raises the `RunError` after its stages and evaluators ran, and its directory keeps their
files but no `summary.json`: the record could not say which files or code it was made with. A
candidate whose values change a `ref` can name an external file the unmodified geometry does not
(`${CAD}/other.stl`): its summary records what that file reads (`candidate_externals`), which
must stay the same while the candidate runs and for every later candidate that names it.
`evaluate(candidate, output, run=...)` records `run`, any JSON-compatible mapping, as the
summary's `run` (a study and trial, say; `null` for `k4opt evaluate`). `identity()` returns what
the session evaluates with: the problem's path and definition, the identities of the files the
stages read, the `implementations` (the version, stage, source and declared metrics of each
evaluator the problem binds, by name, once however often it is bound), the compact file,
source root and fingerprint of the unmodified geometry and what its external references read,
`KEY4HEP_STACK` and `K4GEO`, and k4opt's version, code SHA-256 and checkout. The environment
and k4opt's identity are taken when the session opens, before anything runs, and every summary
of the session records the same k4opt identity, so an edit to the k4opt checkout during a long
run is not recorded as the code that ran (a hash cannot prove which code Python executed, only
narrow the gap). k4opt's code is hashed once per process, since Python runs the modules it
imported; a session does not open, and does not evaluate or record a candidate, once k4opt's
files have changed since then, because no hash would then say which code runs. A session
coordinates its candidates from one thread and is not thread-safe, whatever runs its tasks
([execution](#execution-jobs-and-worker-processes)).

Whatever proposes candidates (a person, an optimizer) can hand them over one by one:

```python
from k4opt.problem import RunRequest

requests = (RunRequest({"constants.ECal_cell_size": v}, f"runs/{v}mm") for v in (5.0, 7.0))
with session.evaluate_each(requests) as evaluations:
    for request, outcome in evaluations:  # outcome: a RunOutcome, or the exception that stopped it
        ...
```

The next request is taken only once the candidates under way leave a job idle, and every
outcome that ended is given back before it; `evaluate()` is one request through the same path.
Close what `evaluate_each` returns (`with`): that stops whatever still runs, which leaving the
loop early does not by itself.

### Execution: jobs and worker processes

Each task of a candidate (its geometry, a stage of a sample, an evaluation) is described as plain
data (its planned task, the files it reads named relative to the candidate's run directory, an
evaluation's request) and done by one task program, wherever it runs; a scheduler decides from
what each task reads which tasks may start. `EvaluationSession.open(..., jobs=N)` (`k4opt
evaluate --jobs N`, and [`k4opt optimize --jobs N`](optimization.md#trials-at-once---jobs) for
the trials of a study) lets up to N tasks run at once, of one candidate (the simulations of its
samples, the evaluations of a simulation) and of several (`evaluate_each`), the oldest
candidate's first:

```python
with EvaluationSession.open(problem, [CaloPhoton], jobs=4) as session:  # stops its workers
    with session.evaluate_each(requests) as evaluations:
        for request, outcome in evaluations:  # in the order they end
            ...
```

- **One job** (the default): every task runs in the session's own thread, one at a time, in
  plan order, with the evaluator instances the candidate's preflight checked; nothing else is
  started.
- **More jobs**: each task runs in a worker process of this machine, started when a task finds
  none idle (one when the session opens, to check that it can work) and kept for the next tasks;
  `close()` (or leaving `with`) stops them, and a later evaluation starts new ones. A worker
  loads the evaluators again by their references and refuses to start (`RunError`) unless they,
  k4opt's code, `KEY4HEP_STACK` and `K4GEO` are the session's. The session itself then only
  checks, assesses and writes the records, and it alone writes them.
- **A job is a processor.** A simulation that ddsim runs on several threads (`numberOfThreads: 4`
  in its stage) holds as many jobs while it runs, so `--jobs 8` runs two such simulations at
  once, or one and four single-threaded tasks. One with more threads than there are jobs holds
  them all and runs alone (the session warns, and says so if it has more threads than the
  processors this process may use, too). While such a simulation waits for its jobs,
  later tasks that fit may start instead, at most N of them, so it never waits for ever; with
  only single-threaded tasks this never happens, and the order is that of one job per task.
  Threads a library starts of its own (ROOT's implicit multi-threading, BLAS) are not counted.
- **`auto`** (`jobs="auto"`, `--jobs auto`) is the processors this process may use: its affinity
  (a Slurm allocation, `taskset`), at most the `cpu.max` quota of its cgroup v2 and of those above
  it (a container, a systemd unit with `CPUQuota`), rounded up. It knows nothing of other users:
  on lxplus it is every core of the node (256), which the node's other users share, so give a
  number there. An HTCondor job without CPU affinity gets the whole machine too: give
  `--jobs $(request_cpus)`. More jobs than usable processors are warned about.
- **N bounds what k4opt starts, not the memory.** Each record holds the peak memory of every task
  (above), and `k4opt evaluate` prints an estimate of what the task that needs most takes while
  it runs. Memory goes with the tasks running at once, not with the jobs: N single-threaded
  simulations need about N times that, but simulations on 4 threads hold 4 jobs each, so at most
  N/4 of them run, and each needs its memory once (its threads share it). Add the session's own
  process. Measure one candidate first, then choose N for the memory as much as for the
  processors, with some room, since it is an estimate; a shared machine (lxplus) is not yours
  alone. k4opt sets no memory limit of its own.
- **Records** are the same whatever N: each task gets the same configuration (geometry, stage
  options and seeds, requests), each record holds its results in plan order with the same
  statuses, blocking and diagnostics, and deterministic evaluators measure the same. Every
  summary records how its tasks ran under `environment.observed.execution` (`{"backend":
  "local", "jobs": 4}`), and each evaluation that ran in a worker its `log`; both are provenance,
  never part of what was evaluated or of any comparison of records.
- **Failures.** An exception of a task (an evaluator's, a contract error) stops its candidate,
  as with one job: it is raised as itself, with the worker's traceback as its cause, and the
  other tasks of the candidate are stopped; the other candidates go on. A worker that dies (a
  segfault, the kernel's OOM killer) fails only its task, naming the signal, and is replaced:
  the candidate is `failed` if that task was required, a diagnostic otherwise, and its record is
  written as for any failed task. The programs a worker started (ddsim) die with it. A worker
  that dies making the geometry, or whose task has killed N + 1 workers in a row (`lower
  --jobs?`), stops its candidate with a `RunError` instead.
- **Stopping.** Ctrl-C reaches only the session, which stops its workers (SIGTERM, then SIGKILL)
  and the simulations they run; a worker also dies when the session's process does (Linux), and
  nothing is published once the evaluations are closed.

- **Watching.** `evaluate_each(requests, activity=...)` calls `activity` in the session's thread
  whenever what runs changes, with an `Activity`: the `candidates` under way and the `busy` of
  the `capacity` job slots (`k4opt optimize` shows them in its progress bar).

A summary is published only after the references only a candidate's geometry names are checked
again against the candidates recorded meanwhile, so two candidates that ran at once never both
get a record that says they read different files.

**Measured** with `tests/benchmarks/parallel.py`, which runs one fixed set of candidates at
several numbers of jobs and checks that every record says what the one-job run says (all but
how it ran). On the fake stack of the tests (8 candidates of two samples, each simulation 2 s of
processor time; lxplus, 2026-11-01):

| jobs | wall s | speedup | efficiency | processors busy | occupancy | records |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 35.4 | 1.00 | 100% | 0.98 | 99% | same |
| 2 | 18.9 | 1.88 | 94% | 1.92 | 95% | same |
| 4 | 10.1 | 3.51 | 88% | 3.69 | 88% | same |
| 8 | 5.7 | 6.26 | 78% | 6.93 | 79% | same |

What is lost goes to starting the workers (each imports k4opt, then podio when it first reads a
file) and to the few candidates at the end, when fewer tasks than jobs are left. Run it with
`--stack real` on the CLD example for the numbers of real simulations, memory included.

**Beyond this machine.** Every task is described as plain data and done by one task program, and
a record verifies without the session that made it, so a batch system can later run tasks, or
whole candidates (each a session with its own jobs on one batch node), without a change to the
problem, the evaluators or the records. Nothing of that exists yet. It would still need the
files moved to where the tasks run and back, the stack and the evaluators' code provided there
and checked, credentials on the batch nodes, and its own memory and processor requests.

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

Each objective value (`ObjectiveValue`) and each limit names its metric: `evaluator`, `sample`
(`None` for an evaluator of the candidate), `stage`, `metric` and their `label`,
`sample/evaluator/stage/metric`. A limit is one bound of a constraint
(`gamma_10gev/calo.photon/simulation/ecal_barrel.angular_residual:max`), judged on
the central value of the measurement: `max` is satisfied when the value is at most the limit,
`min` when it is at least the limit, and equality satisfies both. The uncertainty is recorded
next to the value but does not decide. `margin` is `value - max` or `min - value` in the
metric's unit, zero or below when the limit is satisfied; it is computed exactly and rounded
once, so its sign is always that of the comparison, and it is `null` if it does not fit in a
float (the verdict stays exact). An infeasible candidate is a complete evaluation with real
objective values; it is only outside the limits.

`Assessment.from_summary(spec, summary)` recomputes the assessment of a `summary.json` as read
back: which tasks are required, from the problem; the candidate's status and message from its
required tasks', and its diagnostics from the others; each objective and constraint value from
the own measurements of the evaluation it names. The record's `implementations` must be the evaluators
the problem binds, with its versions and stages; its `samples` the problem's, each with the
evaluations of the bindings that measure it, and its `evaluations` those of the bindings that
measure the candidate, each of the evaluator the problem binds it to, with the `settings` (as `declared`) and `collections` the
problem gives it. It checks again what a live evaluation enforces: the candidate is in the
problem's domains and `candidate_key` is its key, an `invalid` or `failed` evaluator gives a
message, each measurement of a metric its evaluator declares carries the declared unit, a
binding whose evaluator declares metrics declares those named of it, and a constraint that
declares a unit is in the unit of its metric. Each file a task made is named by its reference
and by a path relative to the run directory, in the task's own directory: the output of a stage
that ran and is `done`, with `sha256` `null`, and the artifacts of an evaluation, each with its
SHA-256, those its evaluator declares and, for a `valid` one, every one of them. It raises
`RecordError` if the record is malformed or breaks one of those rules, was made for another
problem (its definition differs, other than in `name`), or if its `status`, `message`,
`required`, `diagnostics`, `represents_candidate`, `metrics` or `assessment` disagree with that
recomputation, so any reader can check a record before
trusting it.

## Deliberately deferred

An evaluator registry or lookup by name (evaluators are named on the command line), chained
combined evaluators, branches of the stages, validating
metric names, `k4run`, hashing stage outputs, a store of artifacts, execution cost, retries, execution on other machines (a
batch system), fingerprints and caching,
persisting or serializing the `EvaluationRecord` apart from `summary.json`, and aggregation,
significance or constraints judged with their uncertainty.
