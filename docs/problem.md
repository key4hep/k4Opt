# Problem specification (draft `k4opt.problem/v1alpha1`)

> **Status: draft.** The file format below can be loaded and statically validated by
> `k4opt.problem` (see [Validation and Python API](#validation-and-python-api)). One
> vertical slice is implemented: `k4opt evaluate` runs a single candidate through its
> stages (a particle gun or an event file, then `ddsim`) once and measures it with each evaluator
> you name, for example the one in `examples/evaluators` (see [evaluators](evaluators.md));
> before it simulates, it checks that every declared objective and constraint is a metric of
> exactly one of them, and afterwards it judges the candidate against the constraints. From
> Python, an `EvaluationSession` evaluates many candidates of a problem the same way
> ([evaluation](evaluation.md#evaluating-candidates-evaluationsession)). Everything else is
> **proposed, not implemented**: no evaluator registry, no `k4run` invocation. An optimizer
> searches the parameters with `k4opt optimize` ([optimization](optimization.md)).
> Parsing a file alone does not check that its evaluators, metrics or geometry targets exist.
> Everything here may change.

k4opt aims to standardize **optimization problems, measurement definitions and evaluation
records**. It does not replace the Key4hep tools that do the work (`ddsim`, `k4run`, ...).
The physicist decides which parameters, ranges, objectives and constraints are
scientifically meaningful; k4opt records them precisely and reproducibly.

**How to read this page.** Everything you write is in `environment`, `geometry`, `parameters`,
`stages`, `evaluators`, `objectives` and `constraints` ([Key reference](#key-reference)). Evaluation
records, provenance and identities ([Concepts](#concepts-and-boundaries-future),
[Identity](#identity-and-provenance)) describe what k4Opt will record and are not part of the file.

The example is [`examples/problem/cld-ecal-granularity.yaml`](../examples/problem/cld-ecal-granularity.yaml),
run with the example evaluator [`calo_photon.py`](../examples/evaluators/calo_photon.py) (see
[evaluators](evaluators.md)). It asks how fine the ECAL barrel cells of CLD should be: finer
cells measure the direction of a photon better, but every photon fires more cells, and each cell
is a readout channel. What a scan of it finds is in [evaluators](evaluators.md#the-example-study).

```yaml
schema_version: k4opt.problem/v1alpha1
name: cld-ecal-granularity

environment:
  key4hep:
    channel: nightly
    version: "2026-10-20"

geometry: FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml

parameters:
  # CLD uses 5.1 mm; 20 mm is about the width of a photon shower
  - geometry:
      constant: ECal_cell_size
    type: continuous
    bounds: [2.5, 20.0]
    unit: mm

# what runs for each candidate: the gun fires inside ddsim, which simulates the photons
stages:
  generation:
    executor: gun
    particle: gamma
    energy:
      value: 10
      unit: GeV
    # the middle of the barrel, where photons enter the ECAL almost head-on
    direction:
      distribution: uniform
      theta: [80, 100]
      phi: [0, 360]
      unit: deg
  simulation:
    executor: ddsim
    events: 1000
    seed: 42
    physics.list: FTFP_BERT  # any ddsim option, named as in `ddsim --help`

# who measures the simulated photons; several evaluators would share one simulation
evaluators:
  - name: calo.photon
    version: "1"
    stage: simulation
    collections:
      hits: ECalBarrelCollection
    settings:
      # a cell fires above half the most probable energy a MIP leaves in 0.5 mm of silicon
      threshold:
        value: 70
        unit: keV

objectives:
  - stage: simulation
    metric: ecal_barrel.angular_residual
    direction: minimize
  - stage: simulation
    metric: ecal_barrel.fired_cells
    direction: minimize
```

## What each part means

Seven things are kept apart; mixing them is the main way two studies silently stop being
comparable.

| Concept | Where | Who decides | Example |
| --- | --- | --- | --- |
| Scientific choices | `parameters`, `objectives`, `constraints` | the physicist | vary `ECal_cell_size` in 2.5-20 mm, minimize the photon's angular residual and the number of fired cells (and, optionally, limit a metric) |
| Stages | `stages`: executors, `events`, `seed`, the gun or a `file` | the physicist, checked by k4opt; shared by every evaluator | 1000 photons of 10 GeV, seed 42 |
| Evaluator inputs | `evaluators[*].collections`, `.settings` | each evaluator's contract, filled by the physicist | the hits in `ECalBarrelCollection`, a threshold of 70 keV |
| Executor options | the other keys of a stage, such as `physics.list` | the physicist, validated by the tool | `physics.list: FTFP_BERT` |
| Execution environment | `environment`, `geometry` | the physicist pins, k4opt resolves | nightly `2026-10-20`, CLD_o2_v09 |
| Measured results | evaluation records | the evaluator | `angular_residual = 0.90 ± 0.03 mrad` |
| Provenance | evaluation records | k4opt records | resolved versions, geometry fingerprint, argv |

The user-facing YAML combines the first five in one file for convenience. Internally they
stay separate (see [Identity](#identity-and-provenance)).

## Key reference

### Top level

| Key | Required | Meaning |
| --- | --- | --- |
| `schema_version` | yes | `k4opt.problem/v1alpha1`. Draft; incompatible changes are expected until a `v1`. |
| `name` | no | Human-readable label of the problem. It carries no semantics and is not an identity; if omitted, the file name is used. |
| `environment` | yes | The software stack the problem is meant to be evaluated with. |
| `geometry` | yes | The starting detector description. |
| `parameters` | yes | The quantities that vary between candidates. |
| `stages` | no | How the data an evaluator reads is made: each stage run by an executor or given as a file. Not needed when no evaluator reads a stage. |
| `evaluators` | yes | Who measures a candidate, and which stage each reads. |
| `objectives` | yes | What to minimize or maximize; one or more. |
| `constraints` | no | Hard limits on measured quantities. |

### `environment`

`environment.key4hep` pins the Key4hep software:

- `channel`: `release` or `nightly`. Both are written as a date (`2026-04-08`, `2026-10-20`), so the
  channel cannot be inferred from `version` and is stated explicitly.
- `version`: the release or nightly identifier (`"2026-10-20"`), always quoted.

The pin identifies an environment; it does not archive its binaries. A nightly may
disappear, so a record must also hold the *resolved* versions (see
[Provenance](#identity-and-provenance)). `environment.key4hep` and `geometry` are not
substitutes: the former selects the software and the installed k4geo, the latter a
compact file inside that k4geo.

### `geometry`

The compact geometry file, relative to the installed k4geo (`$K4GEO`), resolved as
described in [docs/geometry.md](geometry.md#where-the-geometry-comes-from). Resolution is
the job of `k4opt.geometry`, not of this component.

### `parameters`

A list. Each item has a **source** (which thing is varied) and a **kind** (how it varies).

Source. The only source defined so far is `geometry`, which reuses the canonical target
identity of `k4opt.geometry`; no second user-facing name is introduced:

| Draft key | Canonical target |
| --- | --- |
| `geometry: {constant: ECal_cell_size}` | `constants.ECal_cell_size` |

`constant: NAME` is shorthand for the general form `target: constants.NAME`. Any other
canonical target reference from `inspect` (for example
`InnerTrackerBarrel.layer[id=1].rphi_layout.nphi`) is written `geometry: {target: <reference>}`.
The general form is proposed on the same terms as the rest of this draft. Parameter sources other than geometry
(e.g. a generator or reconstruction setting) are possible in the abstract model, but no
syntax is proposed for them here.

Kind. `type` is one of:

| `type` | Required keys | Meaning |
| --- | --- | --- |
| `continuous` | `bounds: [low, high]` | real value, bounds inclusive |
| `integer` | `bounds: [low, high]` | integer value, bounds inclusive |
| `categorical` | `choices: [...]` | one of the listed values |

`bounds` must be two finite numbers with `low < high` (integer parameters: integers; a
boolean or a string is never a number). `choices` must be a non-empty list of unique strings,
or of unique booleans, or of unique finite numbers, not a mixture. `bounds`, `choices` and
`unit` are not interchangeable: `choices` goes with `categorical` only, and a categorical
parameter has no `unit`. A problem lists at least one parameter, and two parameters must not
name the same target (`constant: X` and `target: constants.X` are the same target).

Units are explicit: `unit` names the unit of the numbers in `bounds`. k4Opt never infers
a scientific unit from a textual XML expression such as `5.1*mm`, and it never evaluates
DD4hep expressions. `k4opt evaluate` writes a candidate value followed by its declared unit
into the XML attribute (`6.0` with `unit: mm` becomes `6.0*mm`, a value without `unit` the bare
number); the units it can write are listed in [evaluators](evaluators.md#k4opt-evaluate).
Unit-less parameters (counts, choices) omit `unit`.

Bounds belong to the *search space*. They are distinct from measurement constraints.

For the example: the checked k4geo description has `ECal_cell_size = 5.1*mm`. The constant
is used for the `CartesianGridXY` cell size of both `ECalBarrel_o2_v01_03.xml` and
`ECalEndcap_o2_v01_03.xml`, so changing it alters barrel *and* endcap readouts, even if the
study evaluates only barrel hits.

### `stages`

How the data the evaluators read is made, once per candidate. `stages` maps a stage to how it is
made; its keys are the stage types, in any order:

| Stage | Reads | Run by `executor:` | Or given as `file:` |
| --- | --- | --- | --- |
| `generation` | nothing | `gun`: a particle gun, fired inside ddsim | an event file (HepMC, ...) that ddsim reads |
| `simulation` | `generation`, the candidate geometry | `ddsim` | not while the parameters vary the geometry (see below) |

Each stage has exactly one of `executor` (k4opt runs it for each candidate) and `file` (its output
exists; a relative path is read from the directory of the problem file, and the path, size and
SHA-256 of the file are recorded). A stage given as a file takes no other keys. An executor
takes only its own keys:

- `gun`: the particle gun, `particle`, `energy`, `direction` and an optional `vertex`
  ([ddsim](ddsim.md#particle-gun-stagesgeneration-executor-gun)). It has no `events` or `seed`:
  the gun is fired inside ddsim, makes as many events as the simulation asks for and draws from
  its random numbers, so `events` or `seed` under the gun is an error that points to
  `stages.simulation`.
- `ddsim`: `events`, the number of events to simulate (required), `seed` (optional; ddsim's own
  default otherwise), and any other key is a ddsim option named as `ddsim --help` names it,
  such as `physics.list: FTFP_BERT` ([ddsim](ddsim.md#options-stagessimulation)).

The stages must fit together and fit the evaluators, which loading checks:

- a stage that runs needs the stage it reads, run or given as a file ("simulation reads
  generation: add stages.generation with executor: gun, or give its output as file:");
- every stage an evaluator reads is defined;
- every stage is used, by an evaluator or by a stage that runs (a generation next to a
  simulation given as a file, which needs none, is an error);
- a file cannot stand in for a stage that the parameters change. Every parameter varies the
  geometry, which the simulation reads, so a simulation file was not made with the candidate's
  geometry and every candidate would measure the same events. Only `generation` may be a file
  today. To try an evaluator on an existing simulation, use
  `k4opt evaluate --input simulation=FILE`, which records that the run does not represent the
  candidate ([evaluators](evaluators.md#k4opt-evaluate)).

`events` and `seed` belong to the stages, not to an evaluator, so all evaluators of the problem
measure the same events. They are part of the evaluation context, since a different seed or
event count is a different measurement.

The executor options are a pass-through, with three rules:

1. *Written as ddsim reads them.* k4opt translates each key into the command line according to
   how ddsim reads that kind of option (flag, single value, repeated, several values), in the
   order of the problem, and refuses what it cannot write faithfully
   ([ddsim](ddsim.md#options-stagessimulation)).
2. *Tool-owned validation.* k4opt does not track every option of the installed `ddsim`;
   unknown or invalid names and values are rejected by the tool when it starts.
3. *What an evaluation defines is protected.* The geometry, output, event count, seed and
   primaries come from the problem; an option that would change one of them is an error rather
   than silently winning or losing, and so is a steering file, which ddsim executes as Python
   and which could change anything. A file an option names (`physics.pdgfile`) is recorded with
   its hash, taken before ddsim starts; a file that changes while ddsim runs fails the
   simulation, since the record could not say which content was read.

Options are part of the evaluation context: they can change physics, not only logging, so the
same evaluator name and version with different options is a **different evaluation context**,
and results are not assumed comparable. Choices such as fast versus full simulation change the
validity of the metrics and are never left to a raw option.

Reconstruction and analysis are out of scope for now; they will be further stage types with
their own executors (`k4run`, say), and the problem will name them the same way.

### `evaluators`

Who measures a candidate: one or more evaluators. Each entry:

| Key | Meaning |
| --- | --- |
| `name` | Logical name of a versioned evaluator, e.g. `calo.photon`. It must equal the `name` of an evaluator that `k4opt evaluate --evaluator` loads. Unique within the problem; letters, digits, `.`, `_` and `-`, starting with a letter or digit, because it also names the evaluator's directory in a run. |
| `version` | Version of that evaluator's contract. A string. |
| `stage` | The pipeline stage it reads, one of the `stages`. Required when the evaluator reads one, and absent when it reads none (a cost model working from the parameters alone needs no stage). It must equal the evaluator's own `stage`, so the problem file shows what each evaluator reads. |
| `collections` | The collection of the stage's file that each role the evaluator declares reads, such as `hits: ECalBarrelCollection`. Roles with a default (`particles`, read from `MCParticles`) may be left out. Only with a `stage`. |
| `settings` | Optional. Evaluator-defined, semantically meaningful inputs (a threshold, a cut, ...). |

**One set of stages, several evaluators.** The stages run once per candidate, however many
evaluators read them (see [Stages](evaluation.md#stages)). An evaluator's `collections` and
`settings` hold only what is its own, such as the collection it reads and a threshold; they may
select among the shared events but never change how the events are made. Evaluators that need different samples, event counts or
seeds need separate problems for now; several named samples in one problem are a planned
extension, which is why these keys belong to the stages and not to an evaluator.

**Metrics and their evaluators.** Every objective and constraint metric is measured by exactly
one evaluator. With a single evaluator, that is the one; if it declares its `metrics`, the
objectives and constraints must be among them. With several, each must declare its `metrics`
([evaluation](evaluation.md#writing-an-evaluator)), no metric may be declared by two of them,
and every objective and constraint metric must be declared by one. `k4opt evaluate` checks this,
and that a constraint's `unit` is the unit its evaluator declares, before anything is created.
`stage` on a metric only classifies it: an evaluator may report `analysis` metrics without an
analysis stage. An evaluator reads the files of its stage and of the stages before it.

**Evaluator.** `name` and `version` are the label of a measurement contract. The
evaluator owns the metric definitions and the interpretation of the stage outputs. k4opt ships
none: the physicist writes one (the example is
[`calo_photon.py`](../examples/evaluators/calo_photon.py)) and names it on
the command line, `k4opt evaluate --evaluator FILE.py:Class` (once per evaluator), whose `name`,
`version` and `stage` must equal those of its entry. A registry that resolves the label itself
is deferred.

**Collections.** The roles an evaluator reads and the type of each belong to its contract, and
the roles of its entry are checked against them before anything runs; a collection name the
simulation is not expected to write is a warning, and the simulation's file decides
([evaluation](evaluation.md#collections)). In the example:

| Role | Meaning |
| --- | --- |
| `hits` | the calorimeter hits to measure (`SimCalorimeterHit`), here `ECalBarrelCollection` |
| `particles` | the generated particles (`MCParticle`), `MCParticles` unless named |

**Settings.** The allowed keys of an evaluator's `settings` belong to that evaluator's
contract, and the evaluator checks them before anything runs. A setting with a unit is written
as `value` and `unit`, in any unit of its dimension; the evaluator receives it in the unit it
declares, and the run records both. In the example:

| Key | Meaning |
| --- | --- |
| `threshold` | the energy above which a cell fires (`value: 70`, `unit: keV`); the evaluator receives GeV |

### `objectives`

A non-empty list. Each item:

| Key | Meaning |
| --- | --- |
| `stage` | Scientific classification of the measurement: `simulation` now; `reconstruction` and `analysis` possible later. It is not a request to run a particular pipeline step (that is the evaluator's `stage`), and it does not create separate problem types. |
| `metric` | Metric name defined by the selected evaluator and version, e.g. `ecal_barrel.angular_residual` (measured by `calo.photon`). The part before the dot names what is measured (here the ECAL barrel); collection names appear only in `collections`. |
| `direction` | `minimize` or `maximize`. |

`stage` and `metric` are separate fields, not a combined `simulation.some_metric` string.
Several objectives can be listed. The order has no meaning: it is not a priority and does not
weight anything; it only fixes the order in which results are reported. How objectives are
combined (priorities, weights, scalarization) is deferred.

### `constraints`

Optional hard limits on measured quantities. Each item has `stage` and `metric` as above,
plus `min` and/or `max` (inclusive) and `unit` where the metric has one.

A constraint is judged on the central value of the measurement: `max` is satisfied when the
value is at most the limit, `min` when it is at least the limit, and equality satisfies both.
The uncertainty is recorded next to the value but does not decide (whether it should is an
[unresolved choice](#unresolved-choices)), so a candidate close to a limit is judged by its
central value. Only a `valid` evaluation is judged; an unavailable metric never counts as
satisfied and is never replaced by an invented number. Each run records the verdict and the
margin of every bound, and whether the candidate is feasible
([assessment](evaluation.md#assessment-objectives-limits-and-feasibility)). k4Opt adds no
scientifically motivated constraints of its own: whether a candidate is physically sensible
is the researcher's responsibility. For instance, a requirement on the photon direction:

```yaml
constraints:
  - stage: simulation
    metric: ecal_barrel.angular_residual
    max: 1.0
    unit: mrad
```

A limit such as this needs a baseline measurement before it means anything; in the
[example study](evaluators.md#the-example-study) this one keeps cells up to about 12 mm.

## Metrics are defined by the evaluator

A metric name alone does not make two studies measure the same thing. The evaluator and
version must define each metric precisely: meaning, unit, estimator, event selection and
thresholds, and the uncertainty and statistical conventions. For the example, the evaluator
defines a fired cell (a deposit of at least 70 keV), the measured direction (from the photon's
vertex to the energy-weighted center of the fired cells), the selection (photons that do not
convert in the tracker) and the estimator (the mean, with the standard error of the mean)
([evaluators](evaluators.md)). Neither metric is the whole story: fine cells are chosen above
all to tell nearby showers apart in particle flow, which needs reconstruction.

## Concepts and boundaries (future)

They are named so the YAML sections have a clear home. The evaluator contract (`Evaluator`,
`EvaluationRequest`, `EvaluationResult`, `Measurement`, `EvaluationRecord`) is implemented in
`k4opt.evaluation`, see [evaluation.md](evaluation.md), and `k4opt evaluate` evaluates one
candidate with the evaluators named on the command line; from Python, an `EvaluationSession`
evaluates many, one after another. A registry, a `Run` that records a campaign and a store of
records do not exist yet, and `load_problem()` neither resolves nor runs an evaluator.

| Concept | Role |
| --- | --- |
| `Problem` | The scientific definition: parameters, objectives, constraints. It does not change when the stack or the executor options change. |
| `Evaluator` | A curated, versioned component. Receives a candidate configuration and the outputs of the stages it reads, returns measurements with uncertainties, a status and provenance; the stages run the Key4hep tools. |
| `EvaluationRecord` | One evaluator's result for one candidate in one context. Parameter variations belong here: one unchanged problem evaluates many candidate geometries. |
| `Run` | One execution session producing records, for example one optimization campaign or one manual scan. |

### Evaluation outcomes

Three outcomes are distinguished and are never collapsed into a number:

- **valid**: the evaluation ran and the measurements are scientifically usable;
- **invalid**: the evaluation ran, but the result is not scientifically valid (for
  example, the evaluator's own checks failed);
- **failed**: execution failed (crash, missing output).

Failures are never encoded as zero, infinity or another objective value. Successful
execution alone does not show that a detector variation makes sense; k4Opt does not
adjudicate the user's scientific choices.

## Identity and provenance

Three identities are kept separate, even though the draft YAML shows them in one file:

1. **Problem identity**: parameters (sources, kinds, bounds, units), objectives,
   constraints.
2. **Evaluation context identity**: everything that can change what a measurement means
   for the same candidate: the pinned and resolved Key4hep stack, geometry, the
   evaluators' names, versions, `collections` and `settings`, and the `stages`: their executors, `events`,
   `seed`, the gun or file, and the exact executor options. Changing the nightly or a ddsim
   option does not change the problem, but it **does** change the context.
3. **Run identity**: one execution session.

Four versions identify different things and are never substituted for one another; all
are recorded in provenance:

| Version | What it identifies |
| --- | --- |
| `schema_version` | how the problem YAML is interpreted |
| `environment.key4hep` | the software stack: DD4hep, Geant4, EDM4hep, k4geo, ... |
| `evaluators[*].version` | the definition and implementation of an evaluation procedure and its metrics |
| geometry fingerprint | the exact materialized detector input |

A bare evaluator `version: "1"` is fine for this proposal, but it must eventually resolve to
a precise implementation revision (for example a release of the package that ships the
evaluator), and that revision is what the record stores.

Provenance recorded per evaluation: `k4opt evaluate` writes the geometry snapshot fingerprint
and what its external references read (see [Fingerprint](geometry.md#fingerprint)), each
evaluator's identity, version, stage,
settings and the import-time SHA-256 of the file defining its class, the executed `ddsim`
command (which holds the options in
order), the size and SHA-256 of every given or supplied file and of the files options name, the
k4opt version with the SHA-256 of its code, and the active stack to `summary.json`
([evaluators](evaluators.md#k4opt-evaluate)).
The normalized problem definition is kept in the summary, so later edits to the YAML cannot
change the parameter domains, objectives or constraints recorded for an earlier run.
The resolved versions of the individual Key4hep packages are not recorded yet.

A candidate is checked against the parameter domains above before it is evaluated, by
`k4opt evaluate` and the internal bridge behind it that produces an `EvaluationRecord`; see
[Evaluation contract](evaluation.md#problem-evaluator-record).

## Validation and Python API

`k4opt.problem` turns a file into immutable, typed models. `load_problem(path)` is the
complete loading operation: it validates the YAML **and** resolves its geometry. It locates
`geometry` in the active k4geo tree (`$K4GEO` for a relative path), discovers the
geometry catalog once, including reachable XML includes, and resolves every
`parameters[*].geometry` selector against it. It fails with a clear error if the geometry
cannot be located or read, or if a target is missing, malformed or ambiguous. The result
keeps the discovered geometry and the resolved targets as private state for later reuse;
they take no part in equality, hashing or `to_dict()`, and the current XML values stay raw
text (`5.1*mm`).

`parse_problem(data)` only validates an already-decoded specification. It is pure: it
touches neither the file system nor a geometry tree and needs no Key4hep installation.

```python
from k4opt.problem import load_problem, parse_problem

problem = load_problem("problem.yaml")  # UTF-8 YAML file; geometry and targets resolved
problem.parameters[0].bounds  # (2.5, 20.0)
problem.stages["simulation"].options  # {"physics.list": "FTFP_BERT"}, order preserved
data = problem.to_dict()  # plain JSON-compatible data, a fresh copy each call
assert parse_problem(data) == problem  # parse already-decoded data, no file needed
```

`$K4GEO` must point at the intended installed geometry tree. `environment.key4hep` names the
intended release or nightly, but k4Opt does not select or verify it: the geometry comes from
the active environment, so a resolved target is neither proof that the geometry matches the
pinned stack nor that it is physically valid. The
[CLD example](../examples/problem/cld-ecal-granularity.yaml) loads only where a suitable CLD
geometry is installed.

A target that exists in several reachable files cannot be selected in `v1alpha1`, which has
no per-file disambiguator; loading reports every candidate instead of choosing one.

What is checked, with the location of the first problem in the message
(`parameters[0].bounds: low must be smaller than high, ...`):

- `schema_version` is exactly `k4opt.problem/v1alpha1`; unknown keys are rejected in every
  section the schema defines (not inside an evaluator's `settings`), as are wrong types.
- `stages`: every key is a stage type with exactly one of `executor` (one the stage allows) and
  `file`; the particle gun's keys and the ddsim options are checked as they will be passed on
  ([ddsim](ddsim.md)); `events` (a positive integer, required by ddsim) and `seed` (an integer)
  only where the executor takes them; and the stages fit together and fit the evaluators (see
  [`stages`](#stages)).
- Identifiers (`name`, `entry`, evaluator names and versions, `stage`, `metric`, units, ...)
  are non-empty strings and are never trimmed or changed. `name` is optional and defaults to
  the file name when loaded from a file.
- `evaluators` lists at least one evaluator, each with `name` and `version` and optionally the
  `stage` it reads; names are unique and made of letters, digits, `.`, `_` and `-`, starting
  with a letter or digit. `collections` maps roles to non-empty collection names and needs a
  `stage`. Booleans are not numbers anywhere.
- `settings` may hold any nested data made of `null`, booleans, numbers, strings, lists and
  mappings with string keys. Non-finite numbers, dates, binary data, sets and
  self-referencing YAML aliases are rejected, because they could not be recorded.
- Objectives have a `minimize`/`maximize` direction and at least one is required. A
  constraint has `min` and/or `max`, with `min <= max`. Duplicate parameters, objectives
  (same `stage` and `metric`) and constraints (same `stage` and `metric`) are rejected; an
  objective and a constraint may name the same metric.
- Duplicate YAML mapping keys, non-string keys, YAML tags that construct Python objects,
  invalid UTF-8 and unreadable files are reported as `ProblemConfigError`.

What is not checked: that the Key4hep version is installed or matches the active geometry,
that the evaluator, its settings or the metrics are defined, that units are consistent, that
ddsim knows an option's name and value (it reports that when it starts), and that bounds or
limits make scientific sense.
There is no fingerprint of a problem yet.

## Deferred

Out of scope until later increments: evaluating many candidates from the command line (`k4opt
evaluate` takes one; an `EvaluationSession` takes many from Python); the evaluator registry (the
evaluator is named on the command line) and metric metadata beyond units; reconstruction and
analysis stages and running `k4run`; steering and configuration files (refused for ddsim until
they can be captured); a problem fingerprint; a multi-objective strategy beyond the Pareto
front; non-geometry parameter sources (stage options are named keys so that a parameter can
target one later); several samples per problem, each entering at any stage, and evaluators that
read more than one; branches of the stages; a parameter that drives several geometry targets at
once; example evaluators of metrics that are not simulated measurements (cost or channel-count
models need no new framework: an evaluator without a `stage` can compute them); a named baseline
candidate (an evaluation of the unmodified geometry is the natural baseline for validating
bounds and constraint limits).

## Unresolved choices

- **Value and unit composition.** `unit: mm` is declared next to numeric bounds, while the
  XML holds `5.1*mm`. `k4opt evaluate` writes `<value>*<unit>` for a fixed list of DD4hep
  units; whether more units, or a check against the unit of the original XML expression, are
  needed is open.
- **Quantity notation in `settings` and `stages`.** The convention is a number with an explicit
  `unit` (`energy` with `value: 10` and `unit: GeV`, or a shared `unit` for a group such as
  `theta`/`phi`), never a quoted expression. An evaluator declares such a setting as a
  `Quantity`; energies and lengths are supported, other dimensions are open.
- **Constraint units.** `unit` on a constraint states the unit of `min`/`max`; whether
  the evaluator may convert or must match exactly is undecided.
- **Release `channel` vocabulary** beyond `nightly` and `release`.
- **Constraints and uncertainty.** A constraint such as `angular_residual <= 1 mrad` is judged
  on a measurement with an uncertainty. k4opt compares the central value and records the
  uncertainty next to it, the same for every optimizer. Whether satisfaction should instead be
  required at some confidence is undecided: such a rule would depend on the number of events,
  and it belongs in the problem so that results stay comparable across optimizers.
