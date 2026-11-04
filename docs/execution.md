# Execution

`k4opt.execution` evaluates candidates of a problem end to end: an `EvaluationSession` loads a
problem file once, binds the evaluator implementations to it and evaluates each candidate into a
run directory of its own, where the candidate's geometry is materialized, each pipeline stage of
each sample runs once (ddsim) and each evaluation measures its sample, or the candidate. Its
tasks run in this process or in worker processes; what each candidate's evaluation found is
published as its `summary.json` ([record](record.md)). `k4opt evaluate` evaluates one candidate
from the command line. What an evaluator implements is the [evaluation contract](evaluation.md);
what is evaluated, the [problem](problem.md).

## `k4opt evaluate`

```bash
k4opt evaluate examples/problem/cld-ecal-granularity.yaml \
  --evaluator examples/evaluators/calo_photon.py:CaloPhoton \
  --param ecal_cell_size=10.0 \
  --output runs/cld-ecal-10mm
```

`--evaluator FILE.py:Class` (or `package.module:Class`) names an evaluator; give it once for each
evaluator the problem's `evaluators` name, in any order, however many entries bind it. The file is executed as ordinary Python
([`load_evaluator`](evaluation.md#loading-an-evaluator)), so name only code you trust.

`--input SAMPLE/STAGE=FILE` (repeatable) uses an existing file as the output of a stage of a
sample instead of running it, to try an evaluator on a simulation made earlier, say:
`--input gamma_10gev/simulation=runs/cld-ecal-10mm/samples/gamma_10gev/simulation/events.edm4hep.root`.
The sample is always named, also in a problem with one. The stages of the sample only it would
need do not run either; the other samples run as the problem says. Its path, size and SHA-256
are recorded. A supplied stage that a parameter reaches (the simulation, when the parameters set
the geometry, the gun or a ddsim option) was not made from this candidate's values: the command
says so, and `summary.json` records `represents_candidate: false`, so such a run never counts as
an evaluation of the candidate. One that no parameter reaches (the simulation, when they set
only evaluator settings) is the same for every candidate, and the run represents the candidate:
that says only that nothing the candidate changes was replaced. k4opt identifies the file but
cannot verify that it is the output of the stage the problem defines; the command warns of that,
and the record lists it under `supplied`. Only a record with `supplied: []` is determined by the
problem, the candidate and the identified files alone, which is why a study supplies none.

`--dry-run`, instead of `--output`, checks the candidate as an evaluation checks it before it
simulates (its values, the options of every stage it runs and its files, every evaluator's
settings and `check()`) and shows what each parameter sets, below its value: each attribute of
the geometry, from its value now to the candidate's, each stage option in each sample, each
evaluator setting in each entry of its binding (by the samples the entry measures). No simulation
runs and no output is written, so it is the quick way to see what a parameter changes before an
expensive simulation. It does load the evaluators and calls their `check()`, and it does not show
that the simulation or the evaluation will succeed. In particular, k4opt checks how a ddsim option
is written, not that the installed ddsim knows it: a misspelt `physics.rangecuut` passes the dry
run, and ddsim refuses it when it starts, which fails the simulation. A reconstruction's job is
asked, though: opening the session dry-runs `k4run` with its arguments, so a misspelt argument
of the job refuses the problem before anything runs ([k4run](k4run.md#the-dry-run)). For
[`cld-ecal-bindings.yaml`](../examples/problem/cld-ecal-bindings.yaml), a target of each kind on
two samples:

```text
$ k4opt evaluate examples/problem/cld-ecal-bindings.yaml \
    --evaluator examples/evaluators/calo_photon.py:CaloPhoton \
    --param ecal_cell_size=8.0 --param range_cut=0.5 --param cell_threshold=70 --dry-run
Candidate

ecal_cell_size = 8.0 mm
  geometry: constants.ECal_cell_size (CLD_o2_v09/...xml)
    5.1*mm -> 8.0*mm
range_cut = 0.5 mm
  simulation: physics.rangecut
    gamma_10gev: 0.5*mm
    gamma_50gev: 0.5*mm
cell_threshold = 70.0 keV
  evaluator calo.photon: threshold
    gamma_10gev, gamma_50gev: 70.0 keV

Configuration validated: the candidate's values, the options of every stage it runs, and every evaluator's settings and check().
No simulation executed; no evaluation output created.
```

A candidate it refuses is refused with the message an evaluation would give, and the exit status
is 1; so is one an evaluator [rejects](#what-ends-a-candidate-and-what-stops-a-session), with
nothing created. `--workers` says how a candidate runs, so `--dry-run` refuses it rather than
ignore it. A dry run evaluates nothing, so it checks a problem whose evaluations have time limits
inline too.

**Time limits** are part of the problem ([problem](problem.md#time-limits)): a stage or an
evaluation that runs longer than its `timeout` is stopped and fails, its logs kept, and the
candidate is `failed`. A simulation's limit kills `ddsim` (for a geometry on which Geant4
stalls): the stage fails with the tail of ddsim's standard error, its logs stay in
`samples/<sample>/simulation/`, and the limit is recorded with it
(`samples.<sample>.stages.simulation.timeout`, `null` without one). An evaluation's limit needs
worker processes, which can be stopped: inline, `k4opt evaluate` refuses to evaluate a problem
that limits its evaluations (`--workers 1` evaluates it in one worker).

`--workers N` runs the tasks in N worker processes, up to N at once: the simulations of
different samples, the evaluations of a simulation
([execution](#execution-inline-or-in-worker-processes)); a simulation on several
threads (`numberOfThreads`) holds as many workers' slots, and `--workers auto` takes as many
workers as the processors the command may use. It bounds the processors k4opt keeps busy, not
the memory each task takes: the command prints an estimate of what a task needs while it runs
(`memory:`), and each task's peak is in the summary. Memory goes with the tasks running at once
(at most N, fewer when simulations hold several slots), so choose N for the memory as much as
for the processors. The default, `inline`, runs them one after another in the command's own
process, the easiest to debug. It is not part of the problem, and `summary.json` records it as
provenance only (`environment.observed.execution`).

One problem, one candidate, in an active Key4hep environment (`ddsim`, `podio`, `$K4GEO`); from
Python, an [`EvaluationSession`](#evaluating-candidates-evaluationsession)
evaluates many candidates of a problem the same way.
`--param NAME=VALUE` names a parameter of the problem file and is repeated for each of
them. `--output` must not exist. Before anything is created, the candidate
(keys, types, bounds), the supplied inputs, the evaluators (each one the problem names given
once, with its version and the stage it reads, and no other), the settings and collections of
each binding, the stages of each sample, and the metrics the objectives and constraints name of
each evaluation (see [problem](problem.md#evaluators)) are validated; an error exits with status 1 and creates
nothing. A collection that ddsim is not expected to write for the geometry is printed as a
warning (`k4opt: warning: ...`, with the closest expected name) before anything runs, and so
are two bindings that measure the same twice on a sample and two samples written alike; the run
goes on ([why only a warning](evaluation.md#collections)). Then, in order:

1. the candidate becomes `Override`s (`6.0` with unit `mm` is written `6.0*mm`; units are limited
   to `um mm cm m mrad rad deg MeV GeV TeV`, expressions are not interpreted), and the geometry
   task materializes the geometry into `<output>/geometry` from the catalog that
   `load_problem()` discovered;
2. for each sample, in the order of the problem, the stages its evaluators or roles read are made
   available once each: a generation `file` is checked and hashed, and the `simulation` stage
   runs `ddsim` on that snapshot in `<output>/samples/<sample>/simulation`, and a
   `reconstruction` stage `k4run` on it and the simulation in
   `<output>/samples/<sample>/reconstruction` ([k4run](k4run.md)). Then each evaluator
   that measures the sample, in the order of the problem, reads the files of the sample's stages
   it needs and may write in `<output>/samples/<sample>/evaluators/<binding>`; last, each
   evaluator of the candidate runs in `<output>/evaluators/<binding>`; combined evaluators read
   their role inputs there. Each of
   these is a [task](evaluation.md#stages) that runs only when what it reads succeeded: if a
   stage fails or its output lacks events, the stage is `failed` and the evaluations that read
   it are `blocked`, not called; the others still are, those of the other samples too. An evaluator whose stage's file lacks a
   collection it declares, or holds it with another type, is `failed` without being called: the
   message lists what the file holds;
3. the candidate is [assessed](record.md#assessment-objectives-limits-and-feasibility)
   against the objectives and constraints, and `<output>/summary.json` (`k4opt.run/1`) is
   written. Every path it holds of a file in `<output>` is relative to `<output>`, so the
   directory can be moved or archived
   ([details](evaluation.md#artifacts-and-the-files-of-a-run)):
   - `id`, the record's own identity (a UUID), and `run`, what it was made for: `null` for
     `k4opt evaluate`, a study and trial when an optimizer asks for it;
   - `problem`: the original path and name plus `definition`, the normalized problem loaded
     for this run, including parameter domains, objective directions and constraint limits;
   - `status` and `message` of the candidate ([how they follow from its tasks'](#how-a-candidate-is-evaluated)),
     the `required` tasks that decide them (by label: `geometry`, `gamma_10gev/simulation`,
     `gamma_10gev/calo.photon`), the `diagnostics` that did not succeed (each with its `task`,
     `status` and `message`), the candidate and its key, whether the run
     `represents_candidate`, which stages were
     `supplied` with `--input` (`gamma_10gev/simulation`), and the `warnings` printed before the
     run;
   - `metrics`: every objective and constraint metric, named by its `evaluator` (a binding),
     `sample` (`null` for an evaluator of the candidate), `stage`, `metric` and `label`
     (`gamma_10gev/calo.photon/simulation/ecal_barrel.fired_cells`), and its value, unit and
     uncertainty, which are `null` unless that evaluation's result is `valid`;
   - `assessment`: the objective values in the problem's order, each with its metric and
     direction, each bound of each constraint
     (`limits`, with the value, its margin to the limit and whether it is `satisfied`), and
     whether the candidate is `feasible` (`true`, `false`, or `null` when it is not assessed:
     not `valid`, or not made from the candidate);
   - `implementations`, each evaluator the problem binds once, by name, in the order it first
     binds them: version, the stage it reads, its source identity (`source`: path, size and
     SHA-256 of the file defining its class, taken at import, the SHA-256 of each module it
     imports from beside it (`helpers`, checked before and after each candidate), whether they
     are the bytes `executed`, and the
     `revision` of the git checkout that tracks it;
     [details](evaluation.md#loading-an-evaluator)), the metrics it declares with their units
     (`metrics`, `null` if it declares none) and the names of the artifacts it declares
     (`artifacts`, `null` if it declares none), and its role declarations
     (`roles`, `null` if it declares none);
   - `samples`, each sample of the problem by its name, in problem order, with its `stages` and
     its `evaluations`:
     - `stages`, each once and in pipeline order: its `status` (`done`, `failed` or `blocked`)
       and `message`; the particle gun, or the size and SHA-256 of a given or supplied file; the
       exact `ddsim` command and exit code, the files its options name, with their hashes; the
       events found (`events_read`), its `directory`, its `outputs` (the EDM4hep file of a
       simulation that is `done`: its reference `gamma_10gev/simulation#events`, path and size,
       never hashed, so `sha256` is `null`) and `logs` (`stdout`, `stderr`, `command`), the tasks
       it `reads` (labeled `gamma_10gev/generation`, and `geometry`) and those it was
       `blocked_by`, and the time spent on it (`null` if it did not run);
     - `evaluations`, each binding that measures the sample by its name, in problem order: the
       `implementation` it binds, status, message, settings (`declared` as the problem wrote
       them, `used` as the evaluator received them, with defaults and quantities in the declared
       unit), the collection of each role, the tasks it `reads` (`geometry` and the sample's
       stages) and those it was `blocked_by`, its measurements with uncertainty, the artifacts it
       returned (each with its reference `gamma_10gev/calo.photon#histogram`, path, size and
       SHA-256, taken when it returned) and its provenance, its directory (`workdir`, `null` if it
       was not called) and the time of its `evaluate()` call (`null` if it was not called);
   - `evaluations`: those of the bindings that measure the candidate, with `roles` mapping each
     input role to its ordered producer labels (empty for cost models), otherwise recorded
     the same way;
   - `geometry`: its `status` (always `done`: a geometry that cannot be made stops the run), the
     time spent making it, the geometry fingerprint and manifest, its `changes` (each attribute
     the candidate set: its `parameter`, `target`, `file`, `locator`, `attribute`, `old_value` and
     `new_value`), the parameters that set it (`varies`), the path and SHA-256 of each file its external
     references read (`externals`) and of each file only this candidate's values name
     (`candidate_externals`: a `ref` parameter choosing `${CAD}/other.stl`), the time of the
     whole evaluation, `run_dir`, where the run was made, and the environment (`declared`: the
     problem's pin; `observed`: the k4opt version, `k4opt_source` with the SHA-256 of k4opt's
     own Python files, taken before anything runs, and the commit and dirty flag of its git
     checkout (`null` when it is installed rather than checked out), `KEY4HEP_STACK` and `K4GEO`
     (those the session opened with, checked before each candidate) and the `ddsim` and `k4run`
     that were started; the pin and the stack are not compared).

The command prints the candidate's values, its status, each evaluation's status and metrics with their
uncertainty (a binding named with `as` beside its evaluator, and the sample it measures:
`photon_200kev (calo.photon 1) on gamma_10gev`), each limit with its verdict and margin, whether
the candidate is feasible, the events of each sample, the fingerprint and the paths. For the 10 mm candidate of the
[example study](evaluators.md#the-example-study), with the
[1 mrad constraint](problem.md#constraints) added to the problem (abridged, values rounded):

```text
candidate:   ecal_cell_size = 10.0 mm
status:      valid
evaluator:   calo.photon 1 on gamma_10gev: valid
metric:      simulation/ecal_barrel.angular_residual = 0.946 +- 0.034 [mrad]
limit:       gamma_10gev/calo.photon/simulation/ecal_barrel.angular_residual <= 1.0: satisfied, margin -0.054 [mrad]
feasible:    yes
events:      gamma_10gev: 1000 of 1000
```

The exit status is 0 for a `valid` candidate, feasible or not; otherwise the message (each
required task that is not `done` or `valid`, with its status and reason) goes to stderr and the
exit status is 1. An evaluation that no objective, constraint or required evaluator reads is a
diagnostic ([problem](problem.md#evaluators)): the command lists the diagnostics on stderr before
anything runs, marks each in its line (`calo.occupancy 1 on zz: failed (diagnostic)`) and warns
of each one that did not succeed, but a diagnostic never changes the status or the exit status.

The summary is published only after it has been completely serialized and closed. A failed
write leaves no partial `summary.json`, and an existing file is never overwritten. Filesystem
failures while running a simulation or preparing an evaluator are recorded as failed outcomes;
an unwritable summary is reported as a run error with the run directory for diagnosis.

Several evaluators measure one candidate in the same run and share its simulation. For example,
next to the example evaluator, an evaluator of your own that needs no simulation, say a readout
cost computed from the cell size (`my_cost.py` and `my.readout_cost` are illustrative):

```yaml
# samples as in the example
evaluators:
  - name: calo.photon
    version: "1"
    stage: simulation
    collections:
      hits: ECalBarrelCollection
    settings:
      threshold:
        value: 70
        unit: keV
  - name: my.readout_cost  # reads no stage
    version: "1"

objectives:
  - evaluator: calo.photon  # on the one sample
    stage: simulation
    metric: ecal_barrel.angular_residual
    direction: minimize
  - evaluator: my.readout_cost  # of the candidate
    stage: analysis
    metric: ecal.readout_cost
    direction: minimize
```

```bash
k4opt evaluate problem.yaml \
  --evaluator examples/evaluators/calo_photon.py:CaloPhoton \
  --evaluator my_cost.py:ReadoutCost \
  --param ecal_cell_size=10.0 --output runs/cost-10mm
```

An evaluator that needs other events, another event count or seed measures another sample of the
same problem ([problem](problem.md#samples)); each sample is simulated once per candidate, on the
one geometry.

## Evaluating candidates: `EvaluationSession`

`k4opt.execution.EvaluationSession` evaluates candidates of one problem file,
each into a run directory of its own with a `summary.json`, exactly as `k4opt evaluate` does
for one ([evaluators](#k4opt-evaluate)). It knows no optimizer: a scan by hand, a
script and an optimization library call it the same way, with candidates given as values.

```python
from k4opt.evaluation import load_evaluator
from k4opt.execution import EvaluationSession, Rejected

session = EvaluationSession.open(
    "examples/problem/cld-ecal-granularity.yaml",
    [load_evaluator("examples/evaluators/calo_photon.py:CaloPhoton")],
)
for size in (5.0, 7.0, 10.0):
    outcome = session.evaluate({"ecal_cell_size": size}, f"runs/scan/{size}mm")
    if isinstance(outcome, Rejected):  # an evaluator's check() rejected it: nothing was created
        print(size, outcome)
        continue
    print(size, outcome.candidate.status, outcome.assessment.feasible)
```

| Once, when the session opens | For every candidate, in `check()` and `evaluate()` |
| --- | --- |
| load the problem file (later edits to it do not reach the session) and discover the geometry | check the candidate against the parameter domains |
| bind the evaluator classes by what each class declares, without creating any; check the stages of each sample they read and their settings and collections; decide who measures each metric | build the request of each evaluation for this candidate and call its class's `check` with it, which may reject the candidate; check that each class still declares what it did |
| collect the warnings about collections ddsim is not expected to write, and about what is measured or simulated twice (`session.warnings`); list the diagnostics (`session.diagnostics`, and `session.notice` to tell whoever starts it) | check that every file the stages read, what the geometry's external references read, `KEY4HEP_STACK`, `K4GEO` and k4opt's code are still what they were when the session opened, and that the directories a reconstruction reads hold the same files |
| record the size and SHA-256 of every file the stages read, by where the problem names it (a stage `file`, `samples.gamma_10gev.generation.file`; a file a ddsim option names; a file supplied with `inputs`, `supplied.gamma_10gev.simulation`; a file two samples read is hashed once and named for each), what the geometry's external references read ([`external_files`](geometry.md#fingerprint)), and each evaluator's source identity and declared metrics | `evaluate()` only: materialize the geometry, run the stages, create one instance of each evaluation's evaluator where it runs and call it, check again what the previous row checks, assess, write the summary |

No request or evaluator instance outlives the evaluation it was made for, so an evaluator's
`check()` and `evaluate()` always see that candidate's values, and nothing an instance keeps
reaches another candidate ([lifecycle](evaluation.md#lifecycle-a-class-and-an-instance-per-evaluation)).
`check(candidate)` creates and runs nothing: it raises for what `evaluate()` would refuse
before creating anything, and `CandidateRejected` for a candidate an evaluator rejects, where
`evaluate()` returns a [`Rejected`](#what-ends-a-candidate-and-what-stops-a-session). A file the stages read (named by the
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
([execution](#execution-inline-or-in-worker-processes)).

Whatever proposes candidates (a person, an optimizer) can hand them over one by one:

```python
from k4opt.execution import RunRequest

requests = (RunRequest({"ecal_cell_size": v}, f"runs/{v}mm") for v in (5.0, 7.0))
with session.evaluate_each(requests) as evaluations:
    for request, outcome in evaluations:  # a RunOutcome, a Rejected, or what stopped it
        ...
```

The next request is taken only once the candidates under way leave a slot idle, and every
outcome that ended is given back before it; `evaluate()` is one request through the same path.
Close what `evaluate_each` returns (`with`): that stops whatever still runs, which leaving the
loop early does not by itself.

### Execution: inline or in worker processes

Each task of a candidate (its geometry, a stage of a sample, an evaluation) is described as plain
data (its planned task, the files it reads named relative to the candidate's run directory, an
evaluation's request) and done by one task program, wherever it runs; a scheduler decides from
what each task reads which tasks may start. `EvaluationSession.open(..., workers=...)` (`k4opt
evaluate --workers`, and [`k4opt optimize --workers`](optimization.md#trials-at-once---workers)
for the trials of a study) says where:

```python
with EvaluationSession.open(problem, [CaloPhoton], workers=4) as session:  # stops its workers
    with session.evaluate_each(requests) as evaluations:
        for request, outcome in evaluations:  # in the order they end
            ...
```

- **`"inline"`** (the default of the library and of `k4opt evaluate`): every task runs in the
  session's own thread, one at a time, in plan order; nothing else is started. The evaluators
  are used as they are given, so a class defined in a script, a notebook or a test works. An
  evaluator that crashes the process (a segfault in native code, the kernel's OOM killer) takes
  the session with it, and nothing can stop one that never returns: a problem that limits its
  evaluations (`timeout`) is checked inline but not evaluated (`RunError`; `check_limits()`).
- **N workers** (`k4opt optimize` runs one by default): each task runs in a worker process of
  this machine, up to N at once, of one candidate (the simulations of its samples, the
  evaluations of a simulation) and of several (`evaluate_each`), the oldest candidate's first.
  A worker is started when a task finds none idle (one when the session opens, to check that it
  can work) and kept for the next tasks; `close()` (or leaving `with`) stops them, and a later
  evaluation starts new ones. A worker loads the evaluators again by their references, so they
  must come from `load_evaluator` (`FILE.py:Class` or `module:Class`) and be identified by the
  code that ran (`executed: true`, [details](evaluation.md#loading-an-evaluator)), since a worker
  that loads it from its source must run what the session did; one that was not is refused when
  the session opens (`RunError`), never evaluated inline instead. A
  worker refuses to start unless the evaluators, k4opt's code, `KEY4HEP_STACK` and `K4GEO` are
  the session's. The session itself then only checks, assesses and writes the records, and it
  alone writes them. Even one worker isolates the session from its evaluators: a worker that
  dies fails only its task, until the same task has killed N + 1 workers in a row (below); with
  the one worker of `k4opt optimize`, a second death in a row stops the study.
- **Time limits** ([problem](problem.md#time-limits)): a worker whose task runs longer than its
  `timeout` is stopped (SIGTERM, then SIGKILL after a grace period, then the process groups of
  every program it started and its own) and replaced; the task fails, naming its limit, with its
  logs (`evaluator.log`, ddsim's) kept, and its slots are free again. A stage stops its own
  program at its limit, and its worker 30 s later if k4opt's own work around the program hangs.
  The geometry, k4opt's own work on files, has a fixed limit of 10 minutes; one that runs longer
  is a `failed` geometry (its candidate `failed`, recorded). A time limit reached is never
  counted as a worker's death: a candidate that runs out of time is the candidate's outcome.
  A process an evaluator starts in a session of its own (`setsid`) escapes the process groups
  that are killed.
- **What isolation does not cover.** Each evaluator class's `check` runs in the session's
  process, for every candidate, before anything of the candidate is created, without a time
  limit: that is what rejects a candidate early, and why `check` must stay an inexpensive check
  of the request. `__init__` and `evaluate()` and the stages are what runs in the workers. A
  kill of the session's own process (its OOM, `kill -9`) is not survived either:
  `k4opt optimize --resume` continues the study.
- **A worker is a processor.** A simulation that ddsim runs on several threads
  (`numberOfThreads: 4` in its stage) holds as many workers' slots while it runs, so
  `--workers 8` runs two such simulations at once, or one and four single-threaded tasks. One
  with more threads than there are workers holds them all and runs alone (the session warns,
  and says so if it has more threads than the processors this process may use, too). While such
  a simulation waits for its slots, later tasks that fit may start instead, at most N of them,
  so it never waits for ever; with only single-threaded tasks this never happens, and the order
  is that of one worker per task. Threads a library starts of its own (ROOT's implicit
  multi-threading, BLAS) are not counted.
- **`auto`** (`workers="auto"`, `--workers auto`) is the processors this process may use: its
  affinity (a Slurm allocation, `taskset`), at most the `cpu.max` quota of its cgroup v2 and of
  those above it (a container, a systemd unit with `CPUQuota`), rounded up. It knows nothing of
  other users: on lxplus it is every core of the node (256), which the node's other users share,
  so give a number there. An HTCondor job without CPU affinity gets the whole machine too: give
  `--workers $(request_cpus)`. More workers than usable processors are warned about.
- **N bounds what k4opt starts, not the memory.** Each record holds the peak memory of every task
  (above), and `k4opt evaluate` prints an estimate of what the task that needs most takes while
  it runs. Memory goes with the tasks running at once: N single-threaded simulations need about
  N times that, but simulations on 4 threads hold 4 slots each, so at most N/4 of them run, and
  each needs its memory once (its threads share it). Add the session's own process. Measure one
  candidate first, then choose N for the memory as much as for the processors, with some room,
  since it is an estimate; a shared machine (lxplus) is not yours alone. k4opt sets no memory
  limit of its own (a limit of address space breaks Geant4 and ROOT, which reserve far more
  than they use), but warns when N programs of the stack (ddsim, k4run) at about 2 GiB each
  (CLD's peak at 1.3 to 1.6 GiB) would not fit in the machine's memory, at most its cgroup's
  `memory.max`. A program the kernel kills for want of memory fails its stage with a message
  that names the signal ("was killed by SIGKILL (out of memory? ...)").
- **Records** are the same whatever runs the tasks: each task gets the same configuration
  (geometry, stage options and seeds, requests), each record holds its results in plan order
  with the same statuses, blocking and diagnostics, and deterministic evaluators measure the
  same. Every summary records how its tasks ran under `environment.observed.execution`
  (`{"backend": "inline"}`, or `{"backend": "process", "workers": 4}`), and each evaluation that
  ran in a worker its `log`; both are provenance, never part of what was evaluated or of any
  comparison of records.
- **Failures.** An exception of a task (an evaluator's, a contract error) stops its candidate,
  wherever it ran: it is raised as itself, with the worker's traceback as its cause, and the
  other tasks of the candidate are stopped; the other candidates go on. A worker that dies (a
  segfault, the kernel's OOM killer) fails only its task, naming the signal, and is replaced:
  the candidate is `failed` if that task was required, a diagnostic otherwise, and its record is
  written as for any failed task. The programs a worker started (ddsim) die with it. A worker
  that dies making the geometry, or whose task has killed N + 1 workers in a row (`fewer
  --workers?`), stops its candidate with a `RunError` instead.
- **Stopping.** Ctrl-C reaches only the session, which stops its workers (SIGTERM, then SIGKILL)
  and the simulations they run; a worker also dies when the session's process does (Linux), and
  nothing is published once the evaluations are closed.

- **Watching.** `evaluate_each(requests, activity=...)` calls `activity` in the session's thread
  whenever what runs changes, with an `Activity`: the `candidates` under way and the `busy` of
  the `capacity` slots (`k4opt optimize` shows them in its progress bar).

A summary is published only after the references only a candidate's geometry names are checked
again against the candidates recorded meanwhile, so two candidates that ran at once never both
get a record that says they read different files.

**Measured** with `tests/benchmarks/parallel.py`, which runs one fixed set of candidates at
several numbers of workers and checks that every record says what the run with the fewest says
(all but how it ran). On the fake stack of the tests (8 candidates of two samples, each
simulation 2 s of processor time; lxplus, 2026-11-01, when one meant inline):

| workers | wall s | speedup | efficiency | processors busy | occupancy | records |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 35.4 | 1.00 | 100% | 0.98 | 99% | same |
| 2 | 18.9 | 1.88 | 94% | 1.92 | 95% | same |
| 4 | 10.1 | 3.51 | 88% | 3.69 | 88% | same |
| 8 | 5.7 | 6.26 | 78% | 6.93 | 79% | same |

What is lost goes to starting the workers (each imports k4opt, then podio when it first reads a
file) and to the few candidates at the end, when fewer tasks than workers are left. Run it with
`--stack real` on the CLD example for the numbers of real simulations, memory included.

**Beyond this machine.** Every task is described as plain data and done by one task program, and
a record verifies without the session that made it, so a batch system can later run tasks, or
whole candidates (each a session with its own workers on one batch node), without a change to the
problem, the evaluators or the records. Nothing of that exists yet. It would still need the
files moved to where the tasks run and back, the stack and the evaluators' code provided there
and checked, credentials on the batch nodes, and its own memory and processor requests.

## What ends a candidate, and what stops a session

Everything that can go wrong with a candidate is one of four kinds, and each is reported the
same way by `EvaluationSession`, `k4opt evaluate` and `k4opt optimize`:

| Kind | What produces it | `EvaluationSession` | `k4opt evaluate` | A study |
| --- | --- | --- | --- | --- |
| **rejected** | the `check` of a required evaluation calls `cls.reject("...")` (`CandidateRejected`); a diagnostic's only keeps that diagnostic from running (`invalid`); or the job of a required reconstruction does not take an argument the candidate sets ([k4run](k4run.md#the-dry-run)) | `evaluate()` returns `Rejected(by, message)`; nothing is created | exit 1, nothing created | a FAIL trial marked `rejected`; it goes on |
| **failed** or **invalid** | a stage that fails or runs out of time; a worker that dies (an OOM kill, a crash), its geometry's included; an evaluation that runs out of time, ends with `self.fail()`/`self.invalid()`, or lacks a collection | a `RunOutcome` with that status and a `summary.json` | exit 1 | a FAIL trial with its record; it goes on |
| **contract or framework error** | an evaluator that raises anything else, a `check` that raises anything but `CandidateRejected`, `CandidateRejected` outside `check`, a result that breaks the contract, an `__init__` that raises, a geometry that cannot be materialized, the same task killing N + 1 workers in a row | the exception; no `summary.json` | exit 2 | the trial is FAIL with `k4opt.error`; the study **stops** |
| **provenance** | a file, the environment, k4opt's or an evaluator's code that changed since the session opened, a file a task made that changed, a geometry that is not the one the session resolved (`ProvenanceError`, a `RunError`) | the exception; no `summary.json` | exit 2 | the study **stops** |

A rejected candidate was not evaluated, so it has no record, no measurement and no status of an
evaluation. A failed or invalid one has its record, whose status says why, and never an
invented objective. An error or a provenance change means a bug, or a record that could not say
what was evaluated: nothing is recorded for that candidate, and an optimizer stops rather than
go on with what it cannot trust.

## How a candidate is evaluated

Evaluating a candidate ([`EvaluationSession`](#evaluating-candidates-evaluationsession)), k4opt
in order:

1. validates the candidate (`CandidateError` otherwise): exactly one value per declared
   parameter, keyed by its name; continuous `int`/finite `float` within inclusive bounds;
   integer a real `int` (no `bool`, no rounding) within bounds; categorical one of the choices
   by kind (strings need a string, booleans a boolean, numbers compare numerically). Then it
   configures the candidate: each value written into every target of its parameter (the
   geometry's attributes as overrides, each sample's stages, each entry's settings), as a new
   configuration of this candidate alone, which nothing shared holds. A number a text
   target would round is refused here;
2. matches the evaluator classes to `evaluators` by the `name` each class declares, in any
   order, without creating any: each evaluator the problem names must be given exactly once,
   however many entries bind it, with the declared `version` and reading the declared `stage`,
   and no other (`EvaluationContractError` otherwise, and for a class whose declarations are not
   plain data or whose `check` is not a classmethod). Each entry is
   a binding, identified by its binding name (`as`, its `name` when not given), and each
   evaluation by the binding and the sample it measures (`gamma_10gev/calo.photon`); tasks,
   records, run directories and the objectives and constraints use that identity;
3. before anything runs, checks the stages of each sample the evaluators read and that each
   objective and constraint names a metric its evaluation's evaluator declares, in the declared
   unit (see [problem](problem.md#evaluators); `EvaluationContractError` otherwise), then
   makes the request of each evaluation for the candidate (its settings and collections
   resolved) and calls its evaluator class's `check` with it, which may reject the candidate
   (nothing is then created, and the session gives a `Rejected`). A collection ddsim is not expected to
   write, two bindings that measure the same twice on a sample and two samples written alike are
   warnings, kept with the candidate;
4. runs the candidate's [tasks](evaluation.md#stages), each after the tasks it reads, in this order: the
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
   [Collections](evaluation.md#collections)), or reads a file that changed after its task made it, is
   `failed`; neither is called. `EvaluationFailed` and
   `EvaluationInvalid` become a `failed` or `invalid` result; other exceptions of an evaluator
   propagate;
6. checks each result; a violation is an `EvaluationContractError`, an inconsistent evaluator
   rather than a violated constraint. A `valid` result must report every metric the objectives
   and constraints name of its evaluation (extra metrics are fine) in the `unit` a constraint
   declares; and if the evaluator declares `metrics`, every
   measurement of one, whatever the status, must carry the declared unit. Each artifact must be
   a file in the evaluation's own directory and as the evaluator declares them; it is hashed then
   ([Artifacts](evaluation.md#artifacts-and-the-files-of-a-run)). Whether a value satisfies `min`/`max` is not
   judged here but by the session's
   [assessment](record.md#assessment-objectives-limits-and-feasibility).

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
  process, or the session's own inline) while it did it, all it held included (Python,
  podio, ROOT, the evaluator). It is measured per task: the process's high-water mark is reset
  before each task (`/proc/self/clear_refs`, Linux), so a worker that is reused does not carry
  one task's peak into the next. It is reset before the evaluator's instance is created, so the
  instance's creation counts too, inline and in a worker alike. `null` where the system cannot
  reset it.
- `program`: for a stage that ran a program (ddsim), that program's peak, as the kernel kept it
  when it ended (`wait4`): the largest of ddsim's own and of each process it started and waited
  for (a wrapper script's child), each on its own, never their sum. `null` for every other task,
  and for a program that did not end by itself (killed at its time limit).

`peak_memory(summary)` reads them back, by task label, as `PeakMemory(process, program)`; its
`per_task` is both added: an **estimate** of what the task needs while it runs, however many
slots it holds (a simulation's threads share its memory), not a bound. The
two peaks need not come at once, which makes it high; a program that runs several processes at
once is counted by the largest of them, which makes it low (ddsim is one process, its Geant4
threads share its memory). A true bound of a process tree needs the kernel's accounting of a
cgroup, which k4opt does not set up. `k4opt evaluate` prints the task that needs most
(`memory:      1.93 GiB per task (estimate), for gamma_10gev/simulation (k4opt 312 MiB, ddsim
1.62 GiB)`). Like the times, they say how the candidate ran, never what was evaluated: no
comparison of records reads them.
