# Problem specification (draft `k4opt.problem/v1alpha1`)

> **Status: draft.** The file format below can be loaded and statically validated by
> `k4opt.problem` (see [Validation and Python API](#validation-and-python-api)). One
> vertical slice is implemented: `k4opt evaluate` runs a single candidate through the stages of
> each of its samples (a particle gun or an event file, then `ddsim`) once and measures it with
> each evaluator you name, for example the one in `examples/evaluators` (see [evaluators](evaluators.md));
> before it simulates, it checks that every declared objective and constraint is a metric of the
> evaluation it names, and afterwards it judges the candidate against the constraints. From
> Python, an `EvaluationSession` evaluates many candidates of a problem the same way
> ([evaluation](execution.md#evaluating-candidates-evaluationsession)). Everything else is
> **proposed, not implemented**: no evaluator registry, no analysis stage. An optimizer
> searches the parameters with `k4opt optimize` ([optimization](optimization.md)).
> Parsing a file alone does not check that its evaluators, metrics or geometry targets exist.
> Everything here may change.

k4opt aims to standardize **optimization problems, measurement definitions and evaluation
records**. It does not replace the Key4hep tools that do the work (`ddsim`, `k4run`, ...).
The physicist decides which parameters, ranges, objectives and constraints are
scientifically meaningful; k4opt records them precisely and reproducibly.

**How to read this page.** Everything you write is in `environment`, `geometry`, `parameters`,
`samples`, `evaluators`, `objectives` and `constraints` ([Key reference](#key-reference)). Evaluation
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
  ecal_cell_size:
    type: continuous
    bounds: [2.5, 20.0]
    unit: mm
    # what it varies: the constant ECal_cell_size of the geometry, written as <value>*mm
    targets:
      geometry: ECal_cell_size

# the events each candidate is measured on: one sample of 10 GeV photons; the gun fires inside
# ddsim, which simulates the photons
samples:
  gamma_10gev:
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
| Samples | `samples`: each sample's stages, with their executors, `events`, `seed`, the gun or a `file`, written in the sample or completed from the shared `stages` | the physicist, checked by k4opt; shared by every evaluator that measures the sample | `gamma_10gev`: 1000 photons of 10 GeV, seed 42 |
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
| `timeout` | no | Seconds every stage that runs and every evaluation may take, unless it sets its own `timeout` ([Time limits](#time-limits)). |
| `geometry` | yes | The starting detector description. |
| `parameters` | yes | The quantities that vary between candidates. |
| `stages` | no | Stages every sample starts from, so that samples say only what differs. |
| `samples` | no | The events a candidate is measured on: named samples, each with the stages that make its data, each stage run by an executor or given as a file. Not needed when no evaluator reads a stage. |
| `evaluators` | yes | Who measures a candidate: which stage each reads, of which samples. |
| `objectives` | yes | What to minimize or maximize; one or more. |
| `constraints` | no | Hard limits on measured quantities. |
| `optimization` | no | The levels of a nested optimization: which parameters each searches, and how an inner level picks its best trial ([`optimization`](#optimization)). |

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

`parameters` maps a name to a parameter, in the order the problem lists them. A parameter is a
scientific quantity: its **domain** says how it varies (`type`, `bounds` or `choices`, `unit`),
and its **targets** say which software settings receive its value. The name is what identifies
it everywhere else: a candidate maps each name to a value (`k4opt evaluate --param
ecal_cell_size=6.0`), and records, study reports, exports and the optimizer key the values by it.
It starts with a lowercase letter and holds only lowercase letters, digits and `_`, at most 64
characters. The optimizer searches the domain alone and never sees the targets. An optional
[`title`](#titles) is what figures and the dashboard show instead of the name.

```yaml
parameters:
  ecal_cell_size:
    type: continuous
    bounds: [2.5, 20.0]
    unit: mm
    targets:
      geometry: ECal_cell_size          # the geometry's constant
      evaluators:
        readout: cell_size              # the setting cell_size of the evaluator bound as readout
```

The parameter's name and the names of what it targets belong to different pieces of software,
and k4opt never relates them by their spelling: `ecal_cell_size` (the problem's quantity),
`ECal_cell_size` (a DD4hep constant) and `cell_size` (a setting of the evaluator `readout`) are
the same physical quantity only because the parameter says so.

Domain. `type` is one of:

| `type` | Required keys | Meaning |
| --- | --- | --- |
| `continuous` | `bounds: [low, high]` | real value, bounds inclusive |
| `integer` | `bounds: [low, high]` | integer value, bounds inclusive |
| `categorical` | `choices: [...]` | one of the listed values |

`bounds` must be two finite numbers with `low < high` (integer parameters: integers; a
boolean or a string is never a number). `choices` must be a non-empty list of unique strings,
or of unique booleans, or of unique finite numbers, not a mixture. `bounds`, `choices` and
`unit` are not interchangeable: `choices` goes with `categorical` only, and a categorical
parameter has no `unit`. `unit` names the unit of the numbers in `bounds`; k4Opt never infers a
unit from a textual XML expression such as `5.1*mm`, and never evaluates DD4hep expressions.
Bounds belong to the *search space*: they are distinct from measurement constraints.

Targets. `targets` maps each kind of target to what the value goes to there; a kind takes one
target, or a list of them when the parameter sets several of that kind:

| Key | Target | Written as |
| --- | --- | --- |
| `geometry` | a constant by its name (`ECal_cell_size`), or any other attribute by the canonical reference `k4opt geometry inspect` prints (`InnerTrackerBarrel.layer[id=1].rphi_layout.nphi`) | a DD4hep expression: `6.0*mm` with a `unit`, the bare number without, `true`/`false`, a string as it is |
| a stage, `generation`, `simulation` or `reconstruction` | an option of that stage: the option alone (`physics.rangecut`) in a problem with one sample, otherwise `{option: physics.rangecut, samples: [gamma, pion]}` | as the stage's executor reads the option (below) |
| `evaluators` | a mapping from an evaluator binding (its `as`, or its `name`) to the setting that receives the value, or to a list of them | `{value: v, unit: u}` with a `unit`, as a problem writes a quantity; the value alone without one |

```yaml
targets:
  geometry: [ECal_cell_size, ECal_endcap_cell_size]       # two constants
  simulation:
    - option: physics.rangecut
      samples: [gamma]
    - option: part.minimalKineticEnergy
      samples: [pion]
  evaluators:
    readout: [cell_size, cell_pitch]                      # two settings of one evaluator
    calo.photon: threshold
```

A stage target names its samples wherever the problem has several: k4opt never guesses which
one a parameter means. An evaluator binding is named whole (`calo.photon`), never split at a dot;
where several entries share the binding on different samples, the setting is set in each.

The options of a stage a parameter can set, and how each is written:

| Executor | Options | Written as |
| --- | --- | --- |
| `gun` (`generation`) | `particle` (a categorical of strings); `energy` (continuous or integer, with a unit of `MeV`, `GeV`, `TeV`) | the string; `{value: v, unit: u}` |
| `ddsim` (`simulation`) | any option that takes one value, named as in `ddsim --help` (`physics.rangecut`, `part.minimalKineticEnergy`) | as ddsim's command line reads it: `0.5*mm` for a number with a unit (DD4hep evaluates it, so `--physics.rangecut=0.5*mm` means what it says), the number, a string or a boolean |
| `k4run` (`reconstruction`) | any argument of the job (`cms`) or Gaudi property (`DDCaloDigi.ECALThreshold`), named as in `k4run OPTIONS.py --help` | the value itself (`--cms=240`), a number, a string or a boolean; the parameter has no `unit`, since the job decides the unit of what it reads and k4opt cannot convert into it |

Not settable, each refused with its reason: `executor`, `file`, `events`, `seed` and `timeout`
(k4opt's own keys of a stage), the gun's `direction` and `vertex`, the options k4opt sets or refuses, a ddsim
option that names a file (`physics.pdgfile`, identified once per session, not per candidate),
`numberOfThreads` (how many processors the simulation keeps busy, not what is simulated), a
flag (ddsim cannot switch one off) and an option that takes a list; for k4run, `options`,
`directory`, `geometry` and `input` (which job runs, identified once per session) and the
arguments k4opt sets. A stage given as a `file:` is not run, so no parameter can set an option
of it. Whether the job takes a value a parameter sets is the job's to say: each end of the
parameter's domain is tried before anything runs (`k4run --dry-run`, below).

An evaluator setting is a target only if the evaluator's implementation **declares** it in its
`settings`, of a kind the parameter's values fit; an evaluator that declares no settings takes
none. That is checked when a session binds the implementations, and a refusal names the
parameter and the target (`parameters.ecal_cell_size.targets.evaluators.readout: setting
'cell_size' expects a quantity in mm, a unit of length (um, mm, cm, m), but the parameter's values
are numbers in GeV`):

| Declared | Takes a parameter whose values are |
| --- | --- |
| `Quantity(unit)` | numbers with a unit of the same dimension (they arrive converted to its unit) |
| `float` | numbers without a unit |
| `int` | integers without a unit |
| `str`, `bool` | strings, booleans |
| `list`, `dict` | never |

An evaluator reads a candidate only through its settings: the request has no parameter values
of its own, so what an evaluator measures depends on no parameter's name.

How targets are checked. A target is set once: two parameters, or one twice, that set the same
attribute, option or setting are refused, also when they spell it differently (`ECal_cell_size`
and `constants.ECal_cell_size` are one attribute; two references a geometry resolves to one
attribute are refused when it is loaded). A target the problem also writes is refused too: the
gun of a sample whose `energy` a parameter sets has no `energy` of its own, nor may the shared
`stages` give it one, and a setting a parameter sets is not in the entry's `settings`. A value
written there would read as used, yet no candidate would use it. So the problem holds what it
declares and nothing else: no value of a parameter, no placeholder. Its stages are checked in two
phases. Each stage is first read on its own; once the parameters are known, its executor checks
it with the options parameters set *pending*: a required one (the gun's `energy`) may be missing,
and every option the stage gives is checked in full. Then each such option is checked at each end
of its parameter's domain (both bounds, or every choice), written in as a candidate would write
it: `bounds: [0, 50]` on an `energy` is refused at `0`, since the gun takes a positive energy.
Every settable option's check decides a value by an interval or a set, so the ends decide every
value between them. Each candidate's stages, complete, are checked again before anything is
created, and `k4opt evaluate --dry-run` shows what a candidate sets
([execution](execution.md#k4opt-evaluate)).

Precision. A candidate's values are exact: its key holds the exact decimal of each number. A
target that reads its value as text (the geometry, a stage option) has it evaluated as a
double, so its parameter's domain must hold only numbers a double holds exactly (integer bounds
within 2⁵³, numeric choices and continuous bounds that are doubles), and a candidate value that
is not one is refused: no record says a value the tool rounded. An evaluator `Quantity` setting
receives a float, the exact value scaled and then rounded once (the record keeps both, as
written and as received); one that would be infinite, or zero or subnormal although it is not
zero, is refused. A `float` or `int` setting receives the value unchanged.

#### What a target changes

Each kind of target changes something else about the experiment, and each is a legitimate thing
to optimize:

- the **geometry** changes the detector itself (a cell size);
- **generation** changes the events measured (a particle's energy);
- **simulation** changes the conditions or the model of the simulation (a range cut);
- an **evaluator setting** changes how an observable is computed from the events (a threshold).

The last is the easiest to misuse. An objective that the varied definition itself moves can be
"optimized" without a better detector: minimizing the number of cells above a threshold while
varying that threshold favors simply raising it. k4opt records what each parameter targets and
what each objective measures, but whether the objectives and constraints make the search
scientifically meaningful is the physicist's to decide;
[`cld-ecal-bindings.yaml`](../examples/problem/cld-ecal-bindings.yaml) varies a threshold and,
for that reason, does not minimize the number of fired cells.

[`cld-ecal-bindings.yaml`](../examples/problem/cld-ecal-bindings.yaml) is a runnable example with
a target of each kind on two samples, with the command that previews a candidate of it.

For the example: the checked k4geo description has `ECal_cell_size = 5.1*mm`. The constant
is used for the `CartesianGridXY` cell size of both `ECalBarrel_o2_v01_03.xml` and
`ECalEndcap_o2_v01_03.xml`, so changing it alters barrel *and* endcap readouts, even if the
study evaluates only barrel hits.

### `samples`

The events a candidate is measured on. `samples` maps a name to a sample, in the order the
problem lists them, which is the order records and reports follow. A sample name is a label:
lowercase letters, digits and `_`, starting with a letter, at most 64 characters
(`gamma_10gev`), so that no two names differ only in case. Objectives, constraints and records
refer to a sample by it, and a run directory holds each sample's work in `samples/<name>`.

A sample maps a stage to how it is made, and may also have `properties` (below) and a
[`title`](#titles). Its stage keys may be in any order:

| Stage | Reads | Run by `executor:` | Or given as `file:` |
| --- | --- | --- | --- |
| `generation` | nothing | `gun`: a particle gun, fired inside ddsim | an event file (HepMC, ...) that ddsim reads |
| `simulation` | `generation`, the candidate geometry | `ddsim` | an EDM4hep file, unless a parameter reaches the stage (see below) |
| `reconstruction` | every event of `simulation`, the candidate geometry | `k4run`: a Gaudi job, such as CLD's reconstruction | an EDM4hep file, unless a parameter reaches the stage |

Each stage has exactly one of `executor` (k4opt runs it for each candidate) and `file` (its output
exists; a relative path is read from the directory of the problem file, and the path, size and
SHA-256 of the file are recorded). A stage given as a file takes no other keys. An executor
takes only its own keys:

- `gun`: the particle gun, `particle`, `energy`, `direction` and an optional `vertex`
  ([ddsim](ddsim.md#particle-gun-generation-executor-gun)). It has no `events` or `seed`: the
  gun is fired inside ddsim, makes as many events as the simulation asks for and draws from its
  random numbers, so `events` or `seed` under the gun is an error that points to the sample's
  `simulation`.
- `ddsim`: `events`, the number of events to simulate (required), `seed`, the seed of its
  random numbers (required: ddsim would draw one of its own, and the stage would make other
  events each time it ran, which its record could not say), and any other key is a ddsim option named as `ddsim --help` names it,
  such as `physics.list: FTFP_BERT` ([ddsim](ddsim.md#options-simulation)).
- `k4run`: the job's `options` files (required; one or a list), an optional `directory` the job
  runs from a copy of (the options files are then named relative to it), optional `geometry`
  and `input` naming the job's own arguments that take the compact file and the input files,
  optional `data` naming the job's arguments that name the data it reads (`{dataFolder:
  /scratch/calibration/}`, passed by k4opt and identified file by file), and any other key is an
  argument of the job, named as `k4run OPTIONS.py --help` names it, such
  as `cms: 91` ([k4run](k4run.md)). It has no `events` or `seed`: the job reconstructs every
  event of the simulation, whose `events` and `seed` made them; a seed the job itself reads is
  one of its arguments.

Every stage that runs (not one given as a `file:`) may take a `timeout`: how many seconds it may
take, the problem's `timeout` when not given ([Time limits](#time-limits)).

Samples share nothing but the candidate's geometry, which every simulation reads, and what the
problem writes once for all of them in [`stages`](#stages). The stages of a sample, once
completed from `stages`, must fit together and fit the evaluators that measure it, which loading
checks:

- a stage that runs needs the stage it reads, run or given as a file ("samples.gamma.simulation:
  simulation reads generation: add samples.gamma.generation with executor: gun, or give its
  output as file:");
- every stage an evaluator reads is defined in each sample it measures;
- every sample is measured by an evaluator, and every stage of a sample is used, by an evaluator
  that measures the sample or by a stage that runs (a generation next to a simulation given as a
  file, which needs none, is an error);
- a file cannot stand in for a stage that a parameter reaches: one that sets an option of the
  stage, or of a stage it reads, or sets the geometry when the stage reads it (the simulation
  does). Such a file was not made from a candidate's values, and every candidate would measure
  the same events. A study whose parameters set only evaluator settings (a threshold) may read a
  fixed simulation file: nothing it stands for depends on the candidate. To try an evaluator on
  an existing simulation, use `k4opt evaluate --input gamma/simulation=FILE`, which records
  whether the run represents the candidate ([evaluators](execution.md#k4opt-evaluate)).

`events` and `seed` belong to a sample's stages, not to an evaluator, so all evaluators of a
sample measure the same events. They are part of the evaluation context, since a different seed
or event count is a different measurement. Each sample is simulated on its own, once per
candidate, also when two samples are written alike; that is a warning, since it simulates the
same conditions twice. A file that several samples read is hashed once and recorded for each.

The executor options are a pass-through, with three rules:

1. *Written as ddsim reads them.* k4opt translates each key into the command line according to
   how ddsim reads that kind of option (flag, single value, repeated, several values), in the
   order of the problem, and refuses what it cannot write faithfully
   ([ddsim](ddsim.md#options-simulation)).
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

A reconstruction is a Gaudi job k4opt runs with `k4run` on the candidate's geometry and every
event of the sample's simulation:

```yaml
reconstruction:
  executor: k4run
  directory: $CLDCONFIG/share/CLDConfig   # runs from a copy of it; $NAME is an environment variable
  options: CLDReconstruction.py           # one file or a list, relative to directory
  geometry: compactFile                   # the job's own argument that takes the compact file
  cms: 91                                 # any argument of the job: --cms=91
  truthTracking: true                     # a flag: --truthTracking
```

Its options follow the same three rules, with what differs for a job that is a Python program
of its own ([k4run](k4run.md)):

1. *Written as k4run reads them.* A key is `--<key>=<value>`, a list `--<key> v1 v2` and `true`
   the bare flag. `false` is refused: a flag of the job's own (`--truthTracking`) takes no value
   and cannot be switched off, and k4opt does not know the job's default; a Gaudi property is
   given the text `'False'`.
2. *The job validates.* The options files define the job's arguments, so k4opt keeps no table of
   them: before anything is simulated, the session runs `k4run --dry-run` with the arguments of
   each reconstruction (each end of the domain of a parameter that sets one), on the problem's
   own geometry and an input of one empty event, and refuses the problem with what k4run said
   (`samples.gamma.reconstruction: the job does not take its arguments: k4run: error: argument
   --cms: invalid choice: '92'`). The ends of a domain do not decide the values between them
   (`--cms` takes four choices), so each candidate that sets an argument is tried too, and
   rejected if the job does not take it ([k4run](k4run.md#the-dry-run)).
3. *What an evaluation defines is protected, and verified.* k4opt gives the job the candidate's
   geometry (`GeoSvc.detectors`, or the job's argument `geometry` names), the simulation's
   events (`IOSvc.Input`, or the argument `input` names), the file to write (`IOSvc.Output`) and
   every event (`--num-events=-1`; CLD's reconstruction sets 3 events itself). An option naming
   one of these, in either of k4run's spellings (`GeoSvc.detectors`, `detectors.GeoSvc`), is an
   error. After the job ran, its resolved options (Gaudi's `JobOptionsSvc` dump) must show that
   it read that geometry and those events and wrote that file, or the reconstruction fails: a
   job such as CLD's, which takes its geometry by an argument of its own and has a default
   (CLD_o2_v07), would otherwise reconstruct the candidate's events on another geometry without
   a word. Every file of `directory` (or each options file) and of what `data` names is
   recorded with its hash, taken
   when the session opens, so that a study refuses to resume with another job; a file that
   changes while k4run runs fails the reconstruction; a file that appears in or goes from those
   directories after the session opened stops the next candidate. An environment variable or a `~` in
   `directory`, `options` or `data` is read when the session opens, too, so every candidate runs
   that job.

The reconstruction's output must hold every event of the simulation it read.

Sample `properties` is an optional mapping of JSON facts for evaluators, for example:

```yaml
properties:
  process: "e+e- -> ZH"
  cross_section: {value: 200, unit: fb}
```

These values reach the sample's evaluator as `request.properties` and combined evaluators as
`RoleInput.properties`. They are immutable copies, recorded in the normalized problem and
compared when a study resumes. k4opt does not interpret units, normalize yields or use these
properties to run stages. Properties alone do not define a sample: it still needs stages, its
own or from `stages`.

### `stages`

Samples often differ in one or two keys, a particle, an energy or a seed, and agree on the rest.
`stages` writes the rest once: it maps a stage to what that stage of every sample starts from.

```yaml
stages:
  generation:
    executor: gun
    particle: gamma
    direction: {distribution: uniform, theta: [80, 100], phi: [0, 360], unit: deg}
  simulation: {executor: ddsim, events: 1000, physics.list: FTFP_BERT}

samples:
  gamma_10gev:
    generation: {energy: {value: 10, unit: GeV}}
    simulation: {seed: 42}
  gamma_50gev:
    generation: {energy: {value: 50, unit: GeV}}
    simulation: {seed: 43}
```

Each sample takes the shared stages as follows:

- a sample that defines no such stage takes the shared one as it is;
- a sample stage without `executor` or `file` completes the shared one key by key: its keys win,
  a nested value (an `energy`, a `direction`) replaces the shared one whole, and nothing is
  removed;
- a sample stage that names `executor` or `file` replaces the shared one; none of its keys are
  taken.

A shared stage names `executor` or `file`, like any stage; a shared `file` takes no other keys,
and a sample cannot complete it, only replace it. Every sample takes every shared stage, so a
sample that does not use one (an event file that nothing simulates, next to a shared
`simulation`) is an error; define that stage in the samples that use it instead. A shared stage
that every sample replaces is an error too, and so is `stages` without `samples`.

A stage is checked once completed, as if it had been written out in the sample. An error names
both places: `samples.gamma_10gev.simulation (completing stages.simulation): missing required key
'events'`, and an invalid key is reported where it was written (`stages.simulation.events` or
`samples.gamma_10gev.simulation.seed`).

`stages` is only a way of writing samples. The normalized problem in records and studies holds
each sample written out in full and no `stages`, so writing the samples out in full or sharing
their stages gives the same problem and the same study.

### `evaluators`

Who measures a candidate: one or more evaluators. Each entry:

| Key | Meaning |
| --- | --- |
| `name` | Logical name of a versioned evaluator, e.g. `calo.photon`. It must equal the `name` of an evaluator that `k4opt evaluate --evaluator` loads. Letters, digits, `.`, `_` and `-`, starting with a letter or digit. |
| `as` | Optional. The name of this binding of the evaluator, `name` when not given. Objectives and constraints name a binding, and so do a run's record and the evaluator's directory in a run, so a binding name means one evaluator, and the names of a problem's bindings differ also when the case of their letters is ignored (a file system may fold it). The same grammar as `name`; not `geometry` or a stage type (`generation`, `simulation`, `reconstruction`): a run's record names those steps beside the evaluators. Needed only to bind an evaluator a second time on a sample. |
| `version` | Version of that evaluator's contract. A string. |
| `stage` | The pipeline stage it reads. Required when the evaluator reads one, and absent when it reads none (a cost model working from the parameters alone needs no stage). It must equal the evaluator's own `stage`, so the problem file shows what each evaluator reads. |
| `samples` | The samples whose `stage` it measures, each on its own: one evaluation per sample, each with its own result. May be left out when the problem has one sample; an evaluator without a `stage` measures the candidate, once, and has none. The problem as k4opt records it always lists them. |
| `collections` | The collection of the stage's file that each role the evaluator declares reads, such as `hits: ECalBarrelCollection`. Roles with a default (`particles`, read from `MCParticles`) may be left out. Only with a `stage`. |
| `roles` | For a combined evaluator: named references to evaluations or stage files of samples. Mutually exclusive with `stage`, `samples` and `collections`; see below. |
| `settings` | Optional. Evaluator-defined, semantically meaningful inputs (a threshold, a cut, ...). |
| `required` | Optional, and only `true`. Makes the candidate's status depend on this binding on each of its samples, as it does on one an objective or constraint names; see below. |
| `timeout` | Optional. Seconds each evaluation of this binding may take, its evaluator's creation included; the problem's `timeout` when not given ([Time limits](#time-limits)). |

**Required evaluations and diagnostics.** The candidate's status is decided by its required
evaluations alone: those whose metrics an objective or a constraint names, those of an entry
marked `required: true`, and everything they read (the stages of their samples, the evaluations
a combined evaluator reads, the geometry). Every other evaluation is a *diagnostic*: it runs once
per candidate like any other, and its result is recorded, but a diagnostic that fails, is
invalid or is blocked leaves the candidate's status, its objectives and its feasibility as they
are. So does a stage that only diagnostics read, such as the simulation of a sample measured by
a diagnostic alone, and a file supplied for it with `--input`, and a diagnostic whose evaluator's
`check` rejects the candidate: it is not run and is recorded `invalid`, where a required
evaluation's rejection rejects the whole candidate. A check meant to veto candidates
without an objective or constraint of its own, an overlap check say, therefore needs
`required: true`:

```yaml
- name: geometry.overlaps
  version: "1"
  required: true        # no objective reads it, but an invalid result must make the candidate invalid
```

`required: false` is refused: an evaluation is already optional unless something needs it.
`k4opt evaluate` and `k4opt optimize` list the diagnostics when they start, and the record of
each candidate lists its `required` tasks and each diagnostic that did not succeed
(`diagnostics`).

**One set of stages per sample, several evaluators.** The stages of a sample run once per
candidate, however many evaluators read them (see [Stages](evaluation.md#stages)). An evaluator's
`collections` and `settings` hold only what is its own, such as the collection it reads and a
threshold; they may select among the events of a sample but never change how the events are
made. Evaluators that need other events, event counts or seeds measure another sample.

**Bindings.** Each entry binds an evaluator to the problem with its own `collections` and
`settings`. One evaluator may be bound several times, each binding with a name of its own
(`as`): two thresholds of `calo.photon` read one simulation and are measured separately. The
evaluator is given once on the command line, however often the problem binds it, and every
evaluation, of every sample, gets an instance of its own. Two bindings with the same
`collections` and `settings` (their defaults filled in) on one sample measure the same twice;
that is a warning, not an error. Two entries may share a binding name to measure other samples
with other settings: they bind the same evaluator, version and stage, on samples that are not
the same.

```yaml
samples:
  gamma_10gev:
    generation: {executor: gun, particle: gamma, energy: {value: 10, unit: GeV},
                 direction: {distribution: uniform, theta: [80, 100], phi: [0, 360], unit: deg}}
    simulation: {executor: ddsim, events: 1000, seed: 42, physics.list: FTFP_BERT}
  gamma_50gev:
    generation: {executor: gun, particle: gamma, energy: {value: 50, unit: GeV},
                 direction: {distribution: uniform, theta: [80, 100], phi: [0, 360], unit: deg}}
    simulation: {executor: ddsim, events: 500, seed: 43, physics.list: FTFP_BERT}

evaluators:
  - name: calo.photon               # one binding, measuring both samples
    version: "1"
    stage: simulation
    samples: [gamma_10gev, gamma_50gev]
    collections: {hits: ECalBarrelCollection}
    settings: {threshold: {value: 70, unit: keV}}
  - name: calo.photon               # the same code, a second configuration on one sample
    as: photon_200kev
    version: "1"
    stage: simulation
    samples: [gamma_10gev]
    collections: {hits: ECalBarrelCollection}
    settings: {threshold: {value: 200, unit: keV}}

objectives:
  - {sample: gamma_10gev, evaluator: calo.photon, stage: simulation,
     metric: ecal_barrel.angular_residual, direction: minimize}
  - {sample: gamma_50gev, stage: simulation,          # the one evaluator of the sample
     metric: ecal_barrel.angular_residual, direction: minimize}
constraints:
  - {evaluator: photon_200kev, stage: simulation,     # the one sample of the evaluator
     metric: ecal_barrel.fired_cells, max: 150, unit: "1"}
```

Each sample is simulated once; there are three evaluations, `gamma_10gev/calo.photon`,
`gamma_10gev/photon_200kev` and `gamma_50gev/calo.photon`.

**Metrics and their evaluations.** Every objective and constraint names the evaluation that
measures its metric: the binding (`evaluator`) and, for an evaluator that reads a stage, the
sample it measures (`sample`). Either may be left out where the problem leaves one choice: the
sample of an evaluator that measures only one, the evaluator of a sample that only one measures,
and both in a problem with one evaluation; an error lists the choices otherwise. A metric of the
same name, of two bindings or of one binding on two samples, makes different references,
labeled `sample/evaluator/stage/metric` (`gamma_10gev/photon_200kev/simulation/ecal_barrel.fired_cells`),
or `evaluator/stage/metric` for an evaluator of the candidate (a `%` or `/` within a part is
percent-escaped, so the stage `analysis/x` with the metric `y` is `e/analysis%2Fx/y` and never
the stage `analysis` with the metric `x/y`; the last two parts are always stage and metric). If the
evaluator declares its `metrics` ([evaluation](evaluation.md#writing-an-evaluator)), the metric
must be among them and a constraint's `unit` must be the unit it declares; `k4opt evaluate`
checks this before anything is created. An evaluator that declares none is checked when its
result arrives. `stage` on a metric only classifies it: an evaluator may report `analysis`
metrics without an analysis stage. An evaluator reads the files of its stage and of the stages
before it.

**Evaluator.** `name` and `version` are the label of a measurement contract. The
evaluator owns the metric definitions and the interpretation of the stage outputs. k4opt ships
none: the physicist writes one (the example is
[`calo_photon.py`](../examples/evaluators/calo_photon.py)) and names it on
the command line, `k4opt evaluate --evaluator FILE.py:Class` (once per evaluator, however many
entries bind it), whose `name`, `version` and `stage` must equal those of each entry that binds
it. A registry that resolves the label itself
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

**Combined evaluators.** A candidate-level binding may read several samples through roles:

```yaml
- name: calo.separation
  as: pi0_rejection
  version: "1"
  roles:
    signal: {sample: gamma_50gev, evaluator: calo.shower_shape}
    background: {sample: pi0_50gev, evaluator: calo.shower_shape}
  settings: {signal_efficiency: 0.9}
```

Each reference has exactly one of `sample` or `samples`, and exactly one of `evaluator` (a
binding name) or `stage`. `samples: [zz, ww]` expands to one input per sample in that order;
a role may also hold a list of references. The normalized problem stores explicit lists of
single-sample references. A sample and binding must exist, and the binding must measure that
sample. Candidate-level bindings cannot be referenced yet. A stage reference must name a
stage that produces a file; gun generation runs inside simulation and has no file of its own.

An item cannot appear twice within a role; it may fill several roles or feed several combined
evaluators. Every producer runs once per candidate. The implementation declares which roles
accept several items and what metrics or artifacts they need; session opening checks these
contracts. A failed, invalid or blocked input blocks the combined evaluator, which is never
called with an incomplete set of inputs. When the combined evaluator is required, what it reads
is required too: an input that failed makes the candidate `failed`, and one that is only invalid
makes it `invalid`.

A combined result belongs to its binding, with no sample, for example
`pi0_rejection/analysis/background_rejection`. The full
[photon/pi0 example](../examples/problem/cld-ecal-pi0-rejection.yaml) runs with gun samples.
Combining process yields for a physics sensitivity needs appropriate input event files and
reconstruction; k4opt performs no implicit arithmetic across samples.

### `objectives`

A non-empty list. Each item:

| Key | Meaning |
| --- | --- |
| `sample` | The sample the metric is measured on, for an evaluator that reads a stage; none for one that measures the candidate. It may be left out when the evaluator measures one sample. |
| `evaluator` | The binding that measures the metric (its `as`, or its `name`). It may be left out when the sample (or, without one, the problem) has one evaluation, and an error lists the choices otherwise. The problem as k4opt records it, in every run and study, always names both. |
| `stage` | Scientific classification of the measurement: `simulation`, `reconstruction`, `geometry` or `analysis`. It is not a request to run a particular pipeline step (that is the evaluator's `stage`), and it does not create separate problem types. |
| `metric` | Metric name defined by the selected evaluator and version, e.g. `ecal_barrel.angular_residual` (measured by `calo.photon`). The part before the dot names what is measured (here the ECAL barrel); collection names appear only in `collections`. |
| `title` | Optional: what figures and the dashboard call the metric ([titles](#titles)). |
| `direction` | `minimize` or `maximize`. |

`stage` and `metric` are separate fields, not a combined `simulation.some_metric` string.
Several objectives can be listed. The order has no meaning: it is not a priority and does not
weight anything; it only fixes the order in which results are reported. How objectives are
combined (priorities, weights, scalarization) is deferred.

### `constraints`

Optional hard limits on measured quantities. Each item has `sample`, `evaluator`, `stage`,
`metric` and an optional `title` as above, plus `min` and/or `max` (inclusive), `unit` where the
metric has one, and `scale` where the problem has constraints on more than one metric (below).

A constraint is judged on the central value of the measurement: `max` is satisfied when the
value is at most the limit, `min` when it is at least the limit, and equality satisfies both.
The uncertainty is recorded next to the value but does not decide (whether it should is an
[unresolved choice](#unresolved-choices)), so a candidate close to a limit is judged by its
central value. Only a `valid` evaluation is judged; an unavailable metric never counts as
satisfied and is never replaced by an invented number. Each run records the verdict and the
margin of every bound, and whether the candidate is feasible
([assessment](record.md#assessment-objectives-limits-and-feasibility)). k4Opt adds no
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

**`scale`.** How far a candidate is beyond a limit is its *margin*, in the metric's unit
(`value - max`, `min - value`; zero or below when satisfied). An optimizer that compares
infeasible candidates adds their margins, which means nothing across metrics of different
units or magnitudes: 2 mrad and 300 cells. `scale` is a characteristic size of the metric, a
positive number in the unit of its limits, and an optimizer is told the margin in units of it,
`margin / scale`, which is dimensionless. Choose it as the amount by which a violation starts
to matter, or the spread of the metric over the search space: it says how a violation of one
constraint weighs against a violation of another, which is a scientific choice the problem
states, never one k4opt makes. Once the constraints name two or more metrics, every constraint
needs a `scale`; with one, it may be left out and counts as 1, which cannot change how its
violations rank. It changes no verdict (a limit is satisfied or not by the physical comparison)
and both the margin and the normalized margin are recorded
([assessment](record.md#assessment-objectives-limits-and-feasibility)). The scales are part of
the problem, so of a study's identity.

```yaml
constraints:
  - {stage: simulation, metric: ecal_barrel.angular_residual, max: 1.0, unit: mrad, scale: 0.2}
  - {evaluator: readout, stage: geometry, metric: ecal_barrel.channels, max: 5.0e+7, unit: "1", scale: 1.0e+6}
```

### `optimization`

Optional: a *nested* optimization, for when the best setting of some parameters depends on others.
For each ECAL cell size, say, the photon finding is tuned anew, and the cell sizes are then compared
each at its best:

```yaml
optimization:
  parameters: [ecal_cell_size]            # the top level searches these
  inner:                                  # for each of its candidates, a study of these:
    parameters: [merge_distance, min_photon_energy]
    trials: 20                            # trials of each such study
    objective: pi0.resolved               # what picks its best trial
```

Each parameter is declared once, under `parameters`, and listed in the `parameters` of exactly one
level. `inner` holds the next level down, and may hold another `inner` in turn. Without `inner`, or
without the block, the problem is an ordinary one: one study searches every parameter.

| Key | Where | Meaning |
| --- | --- | --- |
| `parameters` | every level | The parameters the level searches; the levels above fix theirs. |
| `trials` | inner levels | How many trials each study of the level runs before its best is chosen. It changes what the level above measures (the best of 20 is not the best of 200), so it is part of the problem. The top level's budget is the run's: `k4opt optimize --trials`. |
| `objective` | inner levels | Picks the best (feasible) trial of each study of the level: an objective of the problem, by its metric or its label, or any metric an evaluator of the problem measures, as a mapping with `sample`, `evaluator`, `stage`, `metric` and `direction`, as an objective has. It may be left out when the problem has one objective, which then picks. |

**What judges a candidate at each level.** The top level is judged by the problem's `objectives`
(one or several: a Pareto front) and `constraints`; an inner level by its `objective` and the
problem's `constraints`. A metric only an inner level names stays out of the top level's front:
tune the tracking for its efficiency, say, and compare detectors on their sensitivity and cost.
Every evaluation measures whatever any level names of it; which evaluations are required (decide
the candidate's status) follows from the level's own criteria.

**What runs at each level.** At the top, every evaluation runs, and those its criteria do not read
are diagnostics, recorded but deciding nothing. At a level below, only what its criteria and the
evaluators marked `required: true` read runs, with the stages they read: its studies run many
trials each, and an expensive analysis only the top level reads would otherwise run in every one
of them. In the ECAL study, say, the readout cost only the top level compares is not computed in
the inner trials, while the simulation and reconstruction the photon finding reads are: once per
geometry, every other trial reusing them.

**New events.** The best trial of each inner study is evaluated once more on new events (its seeds
derived, [replicates](execution.md#k4opt-evaluate)) before its level above is told it, since the
best of many trials on the same events looks better than it is. So every sample must make its events
from its seeds: a stage given as a `file` (HepMC generation, a given simulation) is refused under
`inner`.

The problem's levels are checked now; `k4opt optimize` runs nested studies in a later release, and
refuses a problem with `inner` until then.

### Titles

A parameter, a sample, an objective and a constraint may have a `title`: one line of text that
figures, the pipeline graph and the dashboard show instead of the name (or, of an objective or a
constraint, the metric). Units are added to it where a value is shown (`ECAL cell size [mm]`), so
a title holds none. It is plain text: Greek letters and other Unicode characters show everywhere,
LaTeX does not. Without one, the name is shown. The name stays what identifies the thing
everywhere else: `--param`, records, exports and the optimizer never read a title.

```yaml
parameters:
  ecal_cell_size:
    title: ECAL cell size
    # type, bounds, unit, targets as above

samples:
  gamma_10gev:
    title: 10 GeV γ
    # its stages

objectives:
  - sample: gamma_10gev
    metric: ecal_barrel.angular_residual
    title: Angular residual
    direction: minimize
```

A title is one line: a line break, also one at its end (as a YAML `|` block leaves), is refused.
An objective's title names what is measured, not the sample it is measured on: the sample has
a title of its own. Where two objectives (or two constraints) would be called alike, a figure adds
the sample (`Angular residual · 10 GeV γ`), and where even that is alike, the whole label; two
parameters titled alike are each called by the title and the name (`Cell size
(ecal_barrel_size)`). An objective and a constraint on one metric are each called by their own
title ([`names`](visualization.md#names)). A title is no part of the problem's identity: a study
whose problem gained or changed titles resumes, and keeps its records. A study shows the titles
its problem had when the study was made (its `study.json` is written once), also the results a
resume with other titles returns.

## Time limits

A stage that runs or an evaluation that takes longer than its `timeout` (in seconds) is stopped
and fails, and its candidate is `failed` with its logs kept: a simulation on which Geant4
stalls, an evaluator that never returns, cannot hold an unattended campaign. `timeout` may be
given at the top level, for every stage that runs and every evaluation, and on a stage of a
sample (or of the shared `stages`) or on an entry of `evaluators`, which wins:

```yaml
timeout: 1800                      # every stage and evaluation, unless it says otherwise
samples:
  gamma_10gev:
    simulation: {executor: ddsim, events: 1000, seed: 42, timeout: 7200}
evaluators:
  - {name: calo.photon, version: "1", stage: simulation, timeout: 600}
```

The limits are part of the problem, so of a study's identity: a study is judged under one set of
limits, and resuming it with others is refused. How a limit is kept is the executor's
([execution](execution.md#execution-inline-or-in-worker-processes)): a worker process that runs
too long is stopped with every program it started; inline, a stage's program (ddsim) is still
killed at its limit, but nothing stops an evaluation that runs in the session's own process, so a
problem that limits its evaluations needs workers to be evaluated (`--workers 1`; a `--dry-run`
works inline). The geometry is not limited by the problem: k4opt's own work on files, it has a
fixed limit of 10 minutes in a worker.

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
| `Evaluator` | A curated, versioned component. Receives a candidate configuration and the outputs of the stages of a sample it reads, returns measurements with uncertainties, a status and provenance; the stages run the Key4hep tools. |
| `EvaluationRecord` | One evaluator's result for one candidate in one context. Parameter variations belong here: one unchanged problem evaluates many candidate geometries. |
| `Run` | One execution session producing records, for example one optimization campaign or one manual scan. |

### Evaluation outcomes

Three outcomes of an evaluation are distinguished and are never collapsed into a number:

- **valid**: the evaluation ran and the measurements are scientifically usable;
- **invalid**: the evaluation ran, but the result is not scientifically valid (for
  example, the evaluator's own checks failed);
- **failed**: execution failed (crash, missing output, a time limit reached).

A candidate an evaluator's `check` **rejects** is not evaluated at all: nothing of it runs, and
it has none of these outcomes ([execution](execution.md#what-ends-a-candidate-and-what-stops-a-session)).

Only required evaluations and the tasks they read decide the candidate's outcome; a diagnostic
is recorded with its own outcome
([required evaluations and diagnostics](#evaluators)).

Failures are never encoded as zero, infinity or another objective value. Successful
execution alone does not show that a detector variation makes sense; k4Opt does not
adjudicate the user's scientific choices.

## Identity and provenance

Three identities are kept separate, even though the draft YAML shows them in one file:

1. **Problem identity**: parameters (domains and targets), objectives, constraints; not the
   problem's `name` or any [title](#titles), which only name things for a reader.
2. **Evaluation context identity**: everything that can change what a measurement means
   for the same candidate: the pinned and resolved Key4hep stack, geometry, the
   evaluators' names, versions, `collections` and `settings`, and each sample's stages: their
   executors, `events`, `seed`, the gun or file, and the exact executor options. Changing the nightly or a ddsim
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

A candidate is identified by the problem's definition and its key (each parameter's name, kind
and exact value); what it configures, the attribute of the geometry, the stage option and the
setting each value goes to, follows from the two alone, so a record holds it and is checked
against it ([record](record.md)). The geometry fingerprint identifies the geometry only: two
candidates that differ only in a ddsim option or an evaluator setting share it, and differ in
their key and in what they configure.

A bare evaluator `version: "1"` is fine for this proposal, but it must eventually resolve to
a precise implementation revision (for example a release of the package that ships the
evaluator), and that revision is what the record stores.

Provenance recorded per evaluation: `k4opt evaluate` writes the geometry snapshot fingerprint,
each attribute the candidate changed (with its parameter, file, locator and values), what the
geometry's external references read (see [Fingerprint](geometry.md#fingerprint)), each
evaluator's identity, version, stage,
settings and the import-time SHA-256 of the file defining its class, the executed `ddsim`
command (which holds the options in
order), the size and SHA-256 of every given or supplied file and of the files options name, the
k4opt version with the SHA-256 of its code, and the active stack to `summary.json`
([evaluators](execution.md#k4opt-evaluate)).
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
`geometry` target of a parameter against it, each to an attribute of its own. It fails with a clear error if the geometry
cannot be located or read, or if a target is missing, malformed or ambiguous. The result is
the declarative problem alone; an evaluation session keeps the discovered geometry and the
resolved targets, whose current XML values stay raw text (`5.1*mm`).

`parse_problem(data)` only validates an already-decoded specification. It is pure: it
touches neither the file system nor a geometry tree and needs no Key4hep installation.

```python
from k4opt.problem import load_problem, parse_problem

problem = load_problem("problem.yaml")  # UTF-8 YAML file; geometry and targets resolved
problem.parameters[0].bounds  # (2.5, 20.0)
problem.samples["gamma_10gev"].stages["simulation"].options  # {"physics.list": "FTFP_BERT"}
problem.objectives[0].label  # "gamma_10gev/calo.photon/simulation/ecal_barrel.angular_residual"
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
(`parameters.ecal_cell_size.bounds: low must be smaller than high, ...`):

- `schema_version` is exactly `k4opt.problem/v1alpha1`; unknown keys are rejected in every
  section the schema defines (not inside an evaluator's `settings`), as are wrong types.
- `samples`: at least one, each named as above; every key of a sample is a stage type with
  exactly one of `executor` (one the stage allows) and `file`; the particle gun's keys and the
  ddsim options are checked as they will be passed on ([ddsim](ddsim.md)); `events` (a positive
  integer, required by ddsim) and `seed` (an integer) only where the executor takes them; and
  each sample's stages fit together and fit the evaluators that measure it (see
  [`samples`](#samples)).
- Identifiers (`name`, `entry`, evaluator names and versions, `stage`, `metric`, units, ...)
  are non-empty strings and are never trimmed or changed. `name` is optional and defaults to
  the file name when loaded from a file.
- `evaluators` lists at least one evaluator, each with `name` and `version` and optionally the
  `stage` it reads, the `samples` it measures (each once, each a sample of the problem; only with
  a `stage`) and its binding name `as`; names are made of letters, digits, `.`, `_` and `-`,
  starting with a letter or digit; a binding name names one evaluator, version and stage, binds
  a sample once, and differs from the others whatever the case of its letters. `collections`
  maps roles to non-empty collection names and needs a `stage`.
  Booleans are not numbers anywhere.
- `settings` may hold any nested data made of `null`, booleans, numbers, strings, lists and
  mappings with string keys. Non-finite numbers, dates, binary data, sets and
  self-referencing YAML aliases are rejected, because they could not be recorded.
- Objectives have a `minimize`/`maximize` direction and at least one is required. A
  constraint has `min` and/or `max`, with `min <= max`. Each names an evaluation of the problem
  (a binding and the sample it measures), or leaves out what the problem leaves one choice for.
  Duplicate parameters, objectives (same `sample`, `evaluator`, `stage` and `metric`) and
  constraints (the same) are rejected; an objective and a constraint
  may name the same metric.
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
evaluator is named on the command line) and metric metadata beyond units; an analysis stage;
steering files for ddsim (refused until they can be captured, as a reconstruction's directory
is); a problem fingerprint; a multi-objective strategy beyond the Pareto front; samples entering
at any stage, and chained combined evaluators; branches of the stages (truth and pattern-
recognition tracking as two reconstructions of one simulation); a reconstruction of part of the
events, or on several threads (a Gaudi Hive job counts as one processor); a parameter with a
unit that sets a reconstruction option, a nested setting, a file option, `events`, `seed` or
`numberOfThreads`; derived parameters and transforms (a log scale); a named baseline
candidate (an evaluation of the unmodified geometry is the natural baseline for validating
bounds and constraint limits: it would be a candidate like any other, configured as every
candidate is, which is why a target holds no value of its own); a problem without a
geometry.

## Unresolved choices

- **Value and unit composition.** `unit: mm` is declared next to numeric bounds, while the
  XML holds `5.1*mm`. `k4opt evaluate` writes `<value>*<unit>` for a fixed list of DD4hep
  units; whether more units, or a check against the unit of the original XML expression, are
  needed is open.
- **Quantity notation in `settings` and stages.** The convention is a number with an explicit
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
