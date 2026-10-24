# Problem specification (draft `k4opt.problem/v1alpha1`)

> **Status: draft.** The file format below can be loaded and statically validated by
> `k4opt.problem` (see [Validation and Python API](#validation-and-python-api)). Everything
> else is **proposed, not implemented**: no evaluator, no metric, no registry, no
> `ddsim`/`k4run` invocation, no optimizer, and no `k4opt problem` CLI command. Parsing a
> file does not check that its evaluator, metrics or geometry targets exist. Everything
> here may change.

k4opt aims to standardize **optimization problems, measurement definitions and evaluation
records**. It does not replace the Key4hep tools that do the work (`ddsim`, `k4run`, ...).
The physicist decides which parameters, ranges, objectives and constraints are
scientifically meaningful; k4opt records them precisely and reproducibly.

**How to read this page.** Everything you write is in `environment`, `geometry`, `parameters`,
`evaluation`, `objectives` and `constraints` ([Key reference](#key-reference)). Evaluation
records, provenance and identities ([Concepts](#concepts-and-boundaries-future),
[Identity](#identity-and-provenance)) describe what k4Opt will record and are not part of the file.

The illustrative example is
[`examples/problem/cld-ecal-granularity.yaml`](../examples/problem/cld-ecal-granularity.yaml).
It is a specification fixture, not an executable study: the evaluator
`calo.single_particle_shower` and both metrics are proposed names, and the bounds (4-8 mm)
and the hit-multiplicity limit (100, a placeholder) have not been validated by a baseline simulation.

```yaml
schema_version: k4opt.problem/v1alpha1
name: cld-ecal-granularity

environment:
  key4hep:
    channel: nightly
    version: "2026-10-20"  # illustrative; availability not verified

geometry:
  entry: FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml

parameters:
  - geometry:
      constant: ECal_cell_size
    type: continuous
    bounds: [4.0, 8.0]
    unit: mm

evaluation:
  evaluator: calo.single_particle_shower
  version: "1"
  events: 1000
  seed: 42
  settings:
    readouts:
      ecal_barrel: ECalBarrelCollection
    generator:
      particle: gamma
      energy:
        value: 10
        unit: GeV
      vertex:
        value: [0, 0, 0]
        unit: mm
      direction:
        distribution: uniform_solid_angle
        theta: [60, 120]
        phi: [0, 360]
        unit: deg
  backend:
    ddsim:
      args: ["--physics.list", "FTFP_BERT"]

objectives:
  - stage: simulation
    metric: ecal_barrel.position_rms
    direction: minimize

constraints:
  - stage: simulation
    metric: ecal_barrel.mean_hit_cells
    max: 100
```

## What each part means

Six things are kept apart; mixing them is the main way two studies silently stop being
comparable.

| Concept | Where | Who decides | Example |
| --- | --- | --- | --- |
| Scientific choices | `parameters`, `objectives`, `constraints` | the physicist | vary `ECal_cell_size` in 4-8 mm, minimize `position_rms` with at most 100 mean hit cells |
| Evaluator settings | `evaluation.events`, `.seed`, `.settings` | the evaluator's contract, filled by the physicist | 1000 photons of 10 GeV, seed 42 |
| Backend arguments | `evaluation.backend.<tool>.args` | the physicist, validated by the tool | `--physics.list FTFP_BERT` |
| Execution environment | `environment`, `geometry.entry` | the physicist pins, k4opt resolves | nightly `2026-10-20`, CLD_o2_v09 |
| Measured results | evaluation records | the evaluator | `mean_hit_cells = 7.9 ± 0.1` |
| Provenance | evaluation records | k4opt records | resolved versions, geometry fingerprint, argv |

The user-facing YAML combines the first four in one file for convenience. Internally they
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
| `evaluation` | yes | How a candidate is measured. |
| `objectives` | yes | What to minimize or maximize; one or more. |
| `constraints` | no | Hard limits on measured quantities. |

### `environment`

`environment.key4hep` pins the Key4hep software:

- `channel`: `release` or `nightly`. Both are written as a date (`2026-04-08`, `2026-10-20`), so the
  channel cannot be inferred from `version` and is stated explicitly.
- `version`: the release or nightly identifier (`"2026-10-20"`), always quoted.

The pin identifies an environment; it does not archive its binaries. A nightly may
disappear, so a record must also hold the *resolved* versions (see
[Provenance](#identity-and-provenance)). `environment.key4hep` and `geometry.entry` are
not substitutes: the former selects the software and the installed k4geo, the latter a
compact file inside that k4geo.

### `geometry`

- `entry`: compact geometry file, relative to the installed k4geo (`$K4GEO`), with the
  same meaning as `geometry` in [docs/geometry.md](geometry.md#where-the-geometry-comes-from).
  Resolution is the job of `k4opt.geometry`, not of this component.

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
DD4hep expressions. How a candidate value plus its declared unit becomes the literal
replacement string for the XML attribute is part of the geometry-resolution increment.
Unit-less parameters (counts, choices) omit `unit`.

Bounds belong to the *search space*. They are distinct from measurement constraints.

For the example: the checked k4geo description has `ECal_cell_size = 5.1*mm`. The constant
is used for the `CartesianGridXY` cell size of both `ECalBarrel_o2_v01_03.xml` and
`ECalEndcap_o2_v01_03.xml`, so changing it alters barrel *and* endcap readouts, even if the
study evaluates only barrel hits.

### `evaluation`

| Key | Meaning |
| --- | --- |
| `evaluator` | Logical name of a curated, versioned standard evaluator, e.g. `calo.single_particle_shower` (proposed). |
| `version` | Version of that evaluator's contract. A string. |
| `events` | Optional. Number of events to process. |
| `seed` | Optional. Random seed. |
| `settings` | Evaluator-defined, semantically meaningful inputs (generator, readouts, ...). |
| `backend` | Raw pass-through arguments for the backend tool(s). |

A problem has exactly one `evaluation`, so every `metric` name resolves against one
evaluator and version, and a candidate has one evaluation context. A study that needs
several stages (say simulation and later reconstruction) uses one evaluator whose contract
covers them and which runs the tools internally; `stage` only classifies the metrics.
Combining several evaluators in one problem is deferred.

**Evaluator.** The default is a standard evaluator referenced by name and version, not an
arbitrary Python path and not a user-written script. The evaluator owns the metric
definitions and the interpretation of backend output. Evaluators are not implemented and
there is no registry yet.

**Events and seed.** `events` and `seed` are common evaluation controls so that different
evaluators do not each invent their own names. Both are optional and only valid where the
evaluator declares support: an evaluator that reconstructs an existing dataset may have no
meaningful seed. They are part of the evaluation context, since a different seed or event
count is a different measurement.

**Settings.** The allowed keys belong to the evaluator's future contract. The keys in the
example are illustrative only. In the example:

| Key | Illustrative meaning |
| --- | --- |
| `readouts` | maps a logical *role* (`ecal_barrel`) to an output collection (`ECalBarrelCollection`). Roles are defined by the evaluator contract, which also says how a collection is turned into hits; add one entry per collection a study needs |
| `generator.particle` | primary particle species (`gamma`) |
| `generator.energy` | primary energy, `value` with an explicit `unit` |
| `generator.vertex` | production point of the primary, `value` (x, y, z) with an explicit `unit`; here the origin. A fixed vertex keeps the true entry point on the ECAL surface well defined, which `position_rms` relies on |
| `generator.direction.distribution` | `uniform_solid_angle`: uniform in cos(theta) and in phi, **not** uniform in theta |
| `generator.direction.theta`, `.phi` | inclusive angular range; `unit` gives the angle unit |

Samples and simulation mode belong in `settings` as well, because they define the
measurement. A study over several physics processes (signal and backgrounds, with event
counts) lists them there; the sample set is not a backend argument. Any sample file or
dataset must be captured for identity (hash or pinned dataset reference), like the files
referenced by backend arguments. Choices such as fast versus full simulation are
evaluator-owned settings too: they change the context identity and the validity of the
metrics, so they are never left to a raw argument.

The 60-120 degree range is an illustrative barrel-oriented selection. It is not shown to
suffice for shower containment or acceptance. The evaluator translates this semantic
request into the concrete backend settings; the user does not write them.

**Backend.** `backend.<tool>.args` is an **ordered list of strings** appended to the
invocation of that tool:

```yaml
backend:
  ddsim:
    args: ["--physics.list", "FTFP_BERT"]
```

Rules (policy for the future implementation; none of it is implemented):

1. *Ordered list, not a string or a mapping.* Order, repeated flags, values and option
   spellings are preserved exactly.
2. *Backend-owned validation.* k4Opt does not track the options of the installed
   `ddsim` or `k4run`; unknown or invalid options are rejected by the tool at execution
   time. The example uses a `ddsim`-style option without k4Opt validating the backend CLI.
3. *No shell.* Arguments are passed as a subprocess argument array with `shell=False`.
   There is no shell parsing, variable substitution, glob expansion or command-string
   evaluation.
4. *Managed inputs are protected.* The evaluator owns the generated command; user
   arguments may extend it but must not silently override what k4Opt manages. Only the
   options k4Opt itself manages are protected, and everything else is passed unchanged to the
   installed tool, which validates it. Some inputs belong to k4Opt or the evaluator: the
   materialized compact geometry, output paths, evaluator-selected steering or job
   configuration, and the authoritative event and generator settings (`events`, `seed` and
   `settings`). A user argument that competes with one of them is an error rather than
   silently winning or losing: the policy is **fail on known conflict**. This is
   best effort. An unrestricted argument list cannot be exhaustively validated, and for
   `k4run` the options are partly defined dynamically by the configuration. Exact conflict
   detection and any merging rules are deferred to the backend integration.
5. *One authoritative way per property.* The example sets the particle gun, event count
   and seed through `events`/`seed` and `settings` and uses raw arguments only for something `settings` does not
   cover (the physics list).
6. *Arguments are part of the evaluation context.* They can change physics, not only
   logging, so the exact ordered list contributes to the context identity. The same
   evaluator name and version with different arguments is a **different evaluation
   context**, and results are not assumed comparable.
7. *Referenced files.* If an argument names a steering file, job-options file, input
   dataset or other external content, recording the path is not enough. The implementation
   should record file identities (hashes) and, where feasible, the resolved effective
   backend configuration. It will not promise to introspect every dynamic `k4run` setting.

Arbitrary backend arguments do not mean arbitrary evaluators: the standard evaluator
still defines the metrics and reads the outputs.

**Later `k4run` counterpart.** Reconstruction is out of scope for now. The same concept
would apply to `k4run` as a sibling key, nothing more:

```yaml
# illustration only; k4run execution is not implemented, and no particular
# option is assumed to exist in every version
backend:
  k4run:
    args: ["..."]      # ordered strings, validated by k4run
```

### `objectives`

A non-empty list. Each item:

| Key | Meaning |
| --- | --- |
| `stage` | Scientific classification of the measurement: `simulation` now; `reconstruction` and `analysis` possible later. It is not a request to run a particular pipeline step, and it does not create separate problem types. |
| `metric` | Metric name defined by the selected evaluator and version, e.g. `ecal_barrel.mean_hit_cells` (proposed). The part before the dot is a role from the evaluator's `readouts` setting, so a metric stays unambiguous when several collections are involved and collection names appear only once, in `settings`. |
| `direction` | `minimize` or `maximize`. |

`stage` and `metric` are separate fields, not a combined `simulation.some_metric` string.
Several objectives can be listed. The order has no meaning: it is not a priority and does not
weight anything; it only fixes the order in which results are reported. How objectives are
combined (priorities, weights, scalarization) is deferred.

### `constraints`

Optional hard limits on measured quantities. Each item has `stage` and `metric` as above,
plus `min` and/or `max` (inclusive) and `unit` where the metric has one.

A constraint is evaluated on the measurement. An unavailable metric never counts as
satisfied and is never replaced by an invented number. Metrics come with
uncertainties, so how a constraint is judged against them is an
[unresolved choice](#unresolved-choices). k4Opt adds no scientifically
motivated constraints of its own: whether a candidate is physically sensible is the
researcher's responsibility. In the example, `max: 100` is a placeholder until a baseline has been measured.

## Metrics are defined by the evaluator

A metric name alone does not make two studies measure the same thing. The evaluator and
version must define each metric precisely: meaning, unit, estimator, event selection and
thresholds, and the uncertainty and statistical conventions. For the example, the
evaluator would have to say exactly what a "hit cell" is (energy threshold, which
collection) and how `position_rms` is computed (reference position, residual definition)
before any number can be interpreted. One possible definition of `position_rms` is the
spread of the shower centroid relative to the true photon intersection with the ECAL
reference surface; this is an example of the precision required, not a decision.

The study as proposed concerns *simulated* calorimeter granularity against a shower-position
observable. Hit multiplicity is not a detector cost or readout channel count, and
`position_rms` is not a validated cost model nor a digitized or reconstructed photon
resolution. The example makes position the objective and multiplicity a limit; a physicist
who wants to minimize multiplicity at a required position resolution swaps the two.

## Concepts and boundaries (future)

Apart from the evaluator *contract*, none of these exist as code yet. They are named so the
YAML sections have a clear home. The contract (`Evaluator`, `EvaluationRequest`,
`EvaluationResult`, `Measurement`) is implemented in `k4opt.evaluation`, see
[evaluation.md](evaluation.md); the evaluator name, metric binding, registry and execution
below are still not, and `load_problem()` neither resolves nor runs an evaluator.

| Concept | Role |
| --- | --- |
| `Problem` | The scientific definition: parameters, objectives, constraints. It does not change when the stack or backend arguments change. |
| `Evaluator` | A curated, versioned component. Receives a candidate configuration and the evaluation context, runs existing Key4hep tools, returns measurements with uncertainties, a status and provenance. |
| `EvaluationRecord` | The result of evaluating one candidate in one context. Parameter variations belong here: one unchanged problem evaluates many candidate geometries. |
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
   for the same candidate: the pinned and resolved Key4hep stack, geometry entry, evaluator
   name and version, `events`, `seed`, `settings`, and the exact ordered backend `args`. Changing the nightly
   or a backend flag does not change the problem, but it **does** change the context.
3. **Run identity**: one execution session.

Four versions identify different things and are never substituted for one another; all
are recorded in provenance:

| Version | What it identifies |
| --- | --- |
| `schema_version` | how the problem YAML is interpreted |
| `environment.key4hep` | the software stack: DD4hep, Geant4, EDM4hep, k4geo, ... |
| `evaluation.version` | the definition and implementation of the evaluation procedure and its metrics |
| geometry fingerprint | the exact materialized detector input |

A bare evaluator `version: "1"` is fine for this proposal, but it must eventually resolve to
a precise implementation revision (for example a release of the package that ships the
evaluator), and that revision is what the record stores.

Provenance recorded per evaluation record (future): resolved Key4hep software versions, the
geometry snapshot fingerprint (see [Fingerprint](geometry.md#fingerprint)), evaluator
identity and version, ordered backend arguments, hashes of relevant input and steering
files, and eventually the executed argv.

A candidate evaluated against a problem is checked against the parameter domains above by an
internal bridge that produces an `EvaluationRecord`; see
[Evaluation contract](evaluation.md#problem-evaluator-record). It is not yet a user-facing
workflow.

## Validation and Python API

`k4opt.problem` turns a file into immutable, typed models. `load_problem(path)` is the
complete loading operation: it validates the YAML **and** resolves its geometry. It locates
`geometry.entry` in the active k4geo tree (`$K4GEO` for a relative entry), discovers the
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
problem.parameters[0].bounds  # (4.0, 8.0)
problem.evaluation.backend["ddsim"]  # ("--physics.list", "FTFP_BERT"), order preserved
data = problem.to_dict()  # plain JSON-compatible data, a fresh copy each call
assert parse_problem(data) == problem  # parse already-decoded data, no file needed
```

`$K4GEO` must point at the intended installed geometry tree. `environment.key4hep` names the
intended release or nightly, but k4Opt does not select or verify it: the geometry comes from
the active environment, so a resolved target is neither proof that the geometry matches the
pinned stack nor that it is physically valid. The illustrative
[CLD example](../examples/problem/cld-ecal-granularity.yaml) loads only where a suitable CLD
geometry is installed.

A target that exists in several reachable files cannot be selected in `v1alpha1`, which has
no per-file disambiguator; loading reports every candidate instead of choosing one.

What is checked, with the location of the first problem in the message
(`parameters[0].bounds: low must be smaller than high, ...`):

- `schema_version` is exactly `k4opt.problem/v1alpha1`; unknown keys are rejected in every
  section the schema defines (not inside `evaluation.settings`), as are wrong types.
- Identifiers (`name`, `entry`, `evaluator`, `stage`, `metric`, units, ...) are non-empty
  strings and are never trimmed or changed. `name` is optional and defaults to the file
  name when loaded from a file.
- `events` is a positive integer and `seed` an integer; booleans are not numbers anywhere.
  Backend names are `ddsim` or `k4run`, each with an `args` list of strings that is stored
  exactly as written.
- `settings` may hold any nested data made of `null`, booleans, numbers, strings, lists and
  mappings with string keys. Non-finite numbers, dates, binary data, sets and
  self-referencing YAML aliases are rejected, because they could not be recorded.
- Objectives have a `minimize`/`maximize` direction and at least one is required. A
  constraint has `min` and/or `max`, with `min <= max`. Duplicate parameters, objectives
  (same `stage` and `metric`) and constraints (same `stage` and `metric`) are rejected; an
  objective and a constraint may name the same metric.
- Duplicate YAML mapping keys, non-string keys, YAML tags that construct Python objects,
  invalid UTF-8 and unreadable files are reported as `ProblemConfigError`.

What is not checked: that the Key4hep version is installed or matches the active geometry, that the evaluator, its settings or the metrics are
defined, that units are consistent, that the backend arguments are valid or conflict with
what the evaluator manages, and that bounds or limits make scientific sense.
There is no fingerprint of a problem yet.

## Deferred

Out of scope until later increments: candidate values and
materialization; CLI commands; the evaluator registry, the binding of `evaluation.evaluator` and metrics to an
implementation, and metric metadata (the bare contract exists, see [evaluation.md](evaluation.md)); running `ddsim` or `k4run`; conflict detection and argument merging;
file and configuration capture; a problem fingerprint; optimizer integration; multi-objective strategy; non-geometry
parameter sources; several evaluators per problem; a parameter that drives several geometry targets at once;
metrics that are not simulated measurements (for example cost or channel-count models);
a named baseline candidate (an evaluation of the unmodified geometry is the natural
baseline for validating bounds and constraint limits); reconstruction and analysis stages.

## Unresolved choices

- **Value and unit composition.** `unit: mm` is declared next to numeric bounds, while the
  XML holds `5.1*mm`. The proposal is that k4Opt composes the replacement string from the
  value and the declared unit; the exact rule is undecided.
- **Quantity notation in `settings`.** The draft convention is a number with an explicit
  `unit` (`energy` with `value: 10` and `unit: GeV`, or a shared `unit` for a group such as
  `theta`/`phi`), never a quoted expression. The evaluator contract may still refine it.
- **Constraint units.** `unit` on a constraint states the unit of `min`/`max`; whether
  the evaluator may convert or must match exactly is undecided.
- **Release `channel` vocabulary** beyond `nightly` and `release`.
- **Constraints and uncertainty.** A constraint such as `mean_hit_cells <= 100` is judged
  on a measurement with an uncertainty. Whether the central value is compared, or
  satisfaction is required at some confidence, is undecided and belongs in the evaluator
  contract so results stay comparable across optimizers.
