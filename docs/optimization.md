# Optimization: `k4opt optimize` and `k4opt study`

`k4opt optimize` runs a **study** of a problem. A *strategy* proposes candidates. Each one is
evaluated exactly as [`k4opt evaluate`](execution.md#k4opt-evaluate) evaluates one, into a
trial directory of the study. The study keeps each trial in its own *ledger*, as that trial's
`summary.json` holds it, read back from disk and checked, and only then tells the strategy. The
default strategy is [Optuna](https://optuna.org)'s TPE sampler; `--strategy grid` runs a grid of
given values instead ([strategies](#strategies-optunas-tpe-and-the-grid)), and
`--strategy FILE.py:Class` [a strategy of your own](#a-strategy-of-your-own). k4opt is built around
none of them: the problem file, the evaluators, the run records, the ledger and everything that reads
a study back know nothing of Optuna, and only one module of this component
(`k4opt/optimization/_optuna.py`) imports it. A study keeps everything it was made with in
`study.json`, and it resumes only with exactly that. `k4opt study` reads a study back, to
inspect and export its results ([below](#inspecting-and-exporting-a-study)), whatever proposed
its candidates.

```bash
k4opt optimize examples/problem/cld-ecal-granularity.yaml \
  --evaluator examples/evaluators/calo_photon.py:CaloPhoton \
  --trials 30 --seed 42 --output runs/cld-ecal-study
```

It prints what the study will do, each trial as it ends, and the study at the end (the values here
are illustrative):

```text
Search ECAL cell size: 30 trials by Angular residual (minimize) and Fired cells (minimize)
at most, if every stage succeeds: simulation 30 · calo.photon 30
trial 0:   valid    feasible    ecal_cell_size=9.79789  gamma_10gev/calo.photon/simulation/ecal_barrel.angular_residual=0.948  gamma_10gev/calo.photon/simulation/ecal_barrel.fired_cells=227
trial 1:   failed   simulation failed: ddsim exited with status 1 (...); blocked: calo.photon  trial-000001/summary.json
...
study:    cld-ecal-granularity-3f9c2a1b, 2 objective(s), 0 limit(s), runs/cld-ecal-study/study.json
trials:   30: 28 valid (28 feasible, 0 infeasible), 1 invalid, 1 failed
pareto:   0, 4, 7, 12, 19
```

A parameter value is shown to 6 significant digits; the trial's record holds all of them. Every
text from the study or its evaluators (names, values, paths, messages, warnings) is printed with
its control characters escaped (`\x1b`, `\n`), so each trial stays one line and nothing an
evaluator reports can change the terminal. On a terminal the lines are colored, and a progress
bar stays under them while the study runs:

```text
⠹ ━━━━━━━━━━━━━━━━━━╸─────────────────────   46%  14/30 trials  1:07:12 elapsed  ~1:16:48 left  4:48/trial  10 feasible  2 infeasible  2 failed
```

It shows the trials asked for that have ended, the time since the study began, the time left,
with several workers the trials running and the slots they hold (`4 running  8/8
slots`), the time a trial takes and how the trials ended; on a narrow terminal the last of these
go first. A trial takes the time since the study began divided by the trials that ended: how
often one ends, which holds when several end together. The time left is that of the trials
still to run, less the time since the last one ended, so it counts down between the ends of
trials. It is an estimate: it assumes the trials to come take as long as those so far. When the study ends, the bar is left in place and marked `✔`, or `✘` when it was stopped;
a study that stops before any of its new trials ended (one that cannot start) leaves only its
error.
Piped or written to a file, the output is the plain lines above, as in a log; `NO_COLOR=1` turns
color off.

| Option | Meaning |
| --- | --- |
| `--evaluator FILE.py:Class` | an evaluator of the problem, as for `k4opt evaluate` (repeatable) |
| `--trials N` | how many *new* trials this invocation runs; each counts, whatever its outcome. A grid with fewer points left ends earlier |
| `--output DIR` | the study directory: new or empty for a new study, the study's own with `--resume` |
| `--strategy optuna\|grid\|FILE.py:Class` | what proposes the candidates: `optuna` (default), Optuna's TPE sampler; `grid`, every combination of given values, in order ([strategies](#strategies-optunas-tpe-and-the-grid)); or a class of your own, `FILE.py:Class` or `package.module:Class`, for a problem without inner levels ([your own](#a-strategy-of-your-own)). A resume keeps the study's, and is given one of its own again |
| `--strategy-settings FILE` | the settings of a strategy of your own, a YAML mapping, passed to it by name and recorded with the study; a resume is given the same |
| `--grid NAME=V1,V2,...` | the values the grid takes for one parameter (repeatable), each read as the parameter's kind, as `--fix` reads one; a continuous parameter needs them, a categorical one takes every choice and an integer one every integer without |
| `--seed S` | the seed of Optuna's sampler, in [0, 2**32); one is drawn and recorded without it. It is not the simulation's seed, which stays each sample's `simulation.seed`; the grid draws nothing and refuses it |
| `--resume` | continue the study in `DIR` (see [Resuming](#resuming-a-study)), with its strategy, seed and grid: give none of `--strategy`, `--grid` and `--seed`, but a strategy of its own and its settings again |
| `--workers N\|auto\|inline` | run the tasks in N worker processes, up to N at once, of one trial and of several; a simulation on several threads holds as many; `auto`: as many as the processors this process may use ([Trials at once](#trials-at-once---workers)). Default 1: one trial after another, each task in turn in one worker, so that an evaluator that crashes or is killed for its memory fails only its trial; if it does so twice in a row (N + 1 worker deaths), the study stops ([execution](execution.md#execution-inline-or-in-worker-processes)), as it would for any task that cannot run. `inline`: in this process, to debug, where such a crash ends the study (`--resume` continues it). Not part of the study: a resume may give another one |
| `--observe files\|libraries\|none` | how closely each stage's program is watched ([software](software.md#what-each-stage-used)). Default `files`: under `strace`, which another study needs to reuse an output. Not part of the study |

The exit status is 0 once every trial asked for has ended, even if candidates were `invalid` or
`failed`. It is 1 if the study cannot start or resume, a trial stops it, or it is interrupted,
and 2 for a usage error. Optuna is not part of the Key4hep stack: `setup.sh` installs the
version pinned in `requirements.txt`, and `pip install 'k4opt[optimize]'` installs it elsewhere.
Only a study of its strategy needs it, to run or resume; a grid study, reading any study back,
publishing and drawing it work without it, as does everything else in k4opt.

Time limits are the problem's (`timeout`, [problem](problem.md#time-limits)), so they are part of
the study: a trial one of whose stages or evaluations runs longer is `failed`,
and the study goes on. A problem that limits its evaluations needs worker processes, which the
default of one worker provides; `--workers inline` refuses it before anything is created.

## Strategies: Optuna's TPE and the grid

A strategy proposes values for the parameters a study searches, and is told how each trial
ended. It does nothing else: k4opt checks each proposal, evaluates it, keeps the trial and its
record, decides when to ask for the next one, and reads the study back. Two strategies come with
k4opt, and you can write [your own](#a-strategy-of-your-own); a study records which one it runs
with under `strategy` in `study.json`, and every study of a [nested
optimization](#nested-studies) runs with the same one, for its own parameters.

**`optuna`**, the default: Optuna's TPE sampler ([below](#optunas-tpe-sampler)). It learns from
every trial that ended, so what it proposes depends on the results and, with several
[workers](#trials-at-once---workers), on the order trials end in.

**`grid`**: every combination of given values of the searched parameters, one after another,
the last parameter varying fastest. It learns nothing, so the same grid proposes the same
candidates in the same order whatever the trials measure, how many run at once and how often the
study is resumed. Scan the ECAL cell size at four values:

```bash
k4opt optimize examples/problem/cld-ecal-granularity.yaml \
  --evaluator examples/evaluators/calo_photon.py:CaloPhoton \
  --strategy grid --grid ecal_cell_size=2.5,5,10,20 \
  --trials 4 --output runs/cld-ecal-grid
```

Before anything runs, it prints the grid under the outline (`grid: 4 points: ecal_cell_size
2.5, 5.0, 10.0, 20.0`; a grid per level for a nested problem).

| Parameter | The grid takes |
| --- | --- |
| `continuous` | the values `--grid` gives, which it needs; a whole number is the float it is (`5` is `5.0`) |
| `integer` | the values `--grid` gives, or every integer between the bounds |
| `categorical` | the values `--grid` gives, or every choice, in the declared order |

Each value is read as the parameter's kind, as `--fix` reads one, and must be in its domain;
none may be given twice, and a fixed parameter takes none. Values with a comma (a categorical
choice `"a,b"`) cannot be given this way; without `--grid` such a parameter takes every choice.
A grid of more than 10 000 points for one study (an integer parameter of a wide range taken
whole, say) is refused before anything is made: each point is a candidate to evaluate. The
Python API takes the same as `optimize(..., strategy="grid", grid={"ecal_cell_size": [2.5, 5.0,
10.0, 20.0]})`.

`--trials N` runs the next N points: a resume goes on from the next point, never proposes one
twice, and a point whose trial was interrupted is spent, not proposed again. Once its last point
is proposed, the study is *exhausted*, whether or not more trials were asked for: the run ends
once its trials ended, `k4opt study show` says so, and the JSON form holds `"exhausted": true`. At an inner level of a nested problem, each study runs the level's `trials` points of its
grid at most, and ends when it is exhausted, with fewer.

`study.json` records the grid as `{"name": "grid", "values": {...}, "points": N}`: the values of
every parameter the study and the levels below it search, and how many points the study's own
grid has; a resume takes it from there, and refuses one that is not a grid of the study, or not
the one the study was made with: the study's ledger commits to it
([below](#how-a-study-is-kept-the-ledger-and-its-strategy)).

### Optuna's TPE sampler

**Parameters.** Each parameter becomes one distribution, named by the parameter's name:

| Parameter | Searched as | Refused when the study is made |
| --- | --- | --- |
| `continuous [low, high]` | uniformly between the bounds, both included; no log scale, no step | a bound that is not exactly a float (`10**400`, `2**53 + 1`), or a range that is not finite |
| `integer [low, high]` | every integer between the bounds | a bound beyond 2**53 in magnitude: the sampler draws integers through floats |
| `categorical` | one of the choices, as declared (string, boolean or number) | nothing |

A proposed candidate is checked against the parameter domains as any other candidate is. Its
values are written into the geometry as they are, with nothing rounded or snapped
(`9.797885082295046*mm`).

**Fixed parameters.** `--fix NAME=VALUE` (repeatable; `optimize(fixed={...})`) gives a parameter
one value for every trial, whatever the strategy: it searches only the others, and every trial's record holds
the complete candidate. Tune an evaluator's settings for one geometry, say: with the geometry
fixed, the study simulates once and every trial reuses that simulation. The fixed values are part
of the study's *scope* (below): a resume needs the same, and a value outside the parameter's
domain, an unknown parameter, or every parameter fixed is refused before anything is created.

**Objectives.** A strategy is told each objective's value, in the problem's order (each with its
`direction`, which the problem and `study.json` record). One objective and several take the same
path. Each is named by its label,
`sample/evaluator/stage/metric`, with the sample and the binding that measures it
(`gamma_10gev/calo.photon/simulation/ecal_barrel.fired_cells`; `evaluator/stage/metric` for an
evaluator of the candidate; a `%` or `/` within a part is percent-escaped, so two objectives
never share a label). The values are those of the run's
[assessment](record.md#assessment-objectives-limits-and-feasibility): that evaluation's
measurement, as a float. A value that no float holds exactly (an integer beyond 2**53) stops
the study rather than being told as another value.

**Constraints.** Each bound of a constraint is a *limit*, named by its metric's label and its
bound, `sample/evaluator/stage/metric:bound`
(`gamma_10gev/calo.photon/simulation/ecal_barrel.angular_residual:max`). For each limit a
strategy is told its *normalized margin*: the physical margin (`value - max`, `min - value`, in
the metric's unit) divided by the constraint's [`scale`](problem.md#constraints), a
dimensionless number that is zero or below when the limit is satisfied. It is computed exactly
and rounded once, so its sign is always the physical verdict's (a violation too small for a
float is the smallest positive float, never 0); one too large for a float stops the study.
The record keeps both the physical and the normalized margin. A limit is judged on the central value of the
measurement, and its uncertainty is recorded but does not decide. An infeasible candidate is a
complete trial with its real objective values, but it is left out of the Pareto front.

Optuna's TPE handles constraints by ranking infeasible trials by the sum of their positive
values. While feasible trials are scarce (early in a study, or under tight limits), the least
infeasible trials join the "good" trials the sampler learns from. So constraints steer the
search toward feasibility, but they do not guarantee it. With limits on several metrics, the
scales make that sum add like with like: a violation of 2 mrad on a scale of 0.2 mrad weighs as
much as one of 10⁷ channels on a scale of 10⁶. The problem must give them then; with one metric,
only one of its limits can be violated at a time, and its scale (1 by default) changes nothing.

**The sampler.** Optuna's `TPESampler`, made explicitly with its seed, 10 startup trials and
the constant liar, all recorded under `strategy` in `study.json`, so that a change of Optuna's
defaults cannot change a study. It proposes at random until 10 trials have values, and TPE
models the trials from then on. The constant liar matters only for
[trials at once](#trials-at-once---workers): a trial still running counts as one that ended
badly, so the sampler does not propose its neighborhood again before it knows the result.

**Optuna keeps nothing on disk.** When a study opens, its sampler is made in memory from the
study's [ledger](#how-a-study-is-kept-the-ledger-and-its-strategy): every trial, in number
order, as Optuna would hold it (with its values and constraint values, as a failure, or still
running), then seeded for the session. TPE learns from the trials alone, so it proposes what a
sampler that held the study all along would, and an interruption anywhere loses nothing of it.
Samplers that also keep state of their own (CMA-ES, NSGA-II keep theirs in Optuna's trial
attributes) would restart that state on a resume; none of them can be chosen yet.

## A strategy of your own

Any Python class with three methods can propose a study's candidates: a hand-written rule, a
wrapper around SciPy or a Bayesian optimizer, or an agent that asks a language model. k4opt does
all the rest as for its own strategies: it checks each proposal against the parameters' domains,
evaluates it, keeps the trial, decides when to ask, resumes the study and reads it back. The
strategy proposes candidates; it changes nothing of how they are judged, the problem's objectives
and constraints.

[`examples/strategies/compass.py`](../examples/strategies/compass.py) is a complete one, compass
search: it measures the middle of the domain, then all its neighbors one step away at once,
moves to the best of them if it improves, and halves the step otherwise.

```bash
k4opt optimize examples/problem/cld-ecal-granularity.yaml \
  --evaluator examples/evaluators/calo_photon.py:CaloPhoton \
  --strategy examples/strategies/compass.py:CompassSearch \
  --strategy-settings examples/strategies/compass.yaml \
  --trials 12 --workers 2 --output runs/cld-ecal-compass
```

or from Python:

```python
from k4opt.evaluation import load_evaluator
from k4opt.optimization import load_strategy, optimize

report = optimize(
    "examples/problem/cld-ecal-granularity.yaml",
    [load_evaluator("examples/evaluators/calo_photon.py:CaloPhoton")],
    "runs/cld-ecal-compass",
    trials=12,
    workers=2,
    strategy=load_strategy("examples/strategies/compass.py:CompassSearch"),
    strategy_settings={"step": 0.25, "min_step": 0.02},
)
```

### The class

```python
from k4opt.optimization import EXHAUSTED, WAITING, Answer, Feedback, Proposal, StudyContext


class MyStrategy:
    def __init__(self, study: StudyContext, *, step: float = 0.25):  # its settings, by name
        ...

    def propose(self, trial: int) -> Proposal | Answer:  # decide
        return Proposal({"ecal_cell_size": 5.0}, {"why": "the middle"})  # or WAITING, EXHAUSTED

    def proposed(self, trial: int, proposal: Proposal) -> None:  # remember what was kept
        ...

    def tell(self, trial: int, feedback: Feedback) -> None:  # learn how a trial ended
        ...
```

`load_strategy("FILE.py:Class")` (or `"package.module:Class"`) loads and identifies it as
[an evaluator is loaded](evaluation.md#loading-an-evaluator): the file is executed as ordinary
Python, so name only code you trust. It is made for the study as `Class(study, **settings)` and
must say what it needs of the settings by its keyword arguments: one it does not take refuses the
study before anything is created. An optional `exhausted` attribute says, after each proposal, that
it will never propose again; answering `EXHAUSTED` says the same when it is asked.

What it is given, `StudyContext`, read-only:

| Field | |
| --- | --- |
| `study` | the study's id, the same whenever it opens: to name what the strategy keeps elsewhere, or to derive a seed |
| `parameters` | the parameters it searches, in problem order, as the problem declares them: `name`, `kind` (`continuous`, `integer`, `categorical`), `bounds` or `choices`, `unit`, `title`, `targets` |
| `fixed` | the values every trial has (`--fix`) |
| `objectives` | the objectives, each with its `label`, `direction` and `title`, in the order of `Feedback.values` |
| `constraints`, `limits` | the constraints, and the name of each of their limits (`label:min`, `label:max`), as `Feedback.constraints` names them |

What it is told of each trial, `Feedback`:

| Field | |
| --- | --- |
| `status` | `valid`, `invalid`, `failed`, `rejected`, `malformed`, `interrupted`, `error` or `unscored` ([Trials and their outcomes](#trials-and-their-outcomes)) |
| `values` | the objective values of a valid trial, floats, in the order of `study.objectives`; `None` for any other |
| `constraints`, `feasible` | the normalized margin of each limit (zero or below when satisfied) and whether all are satisfied, of a valid trial; nothing for any other |
| `result` | the trial as `load_study` reads it back: its candidate, the assessment of its record (objectives with their unit and uncertainty, each limit with its margin), its diagnostics and times. An invalid or failed trial has a record, assessed without objectives; a rejected, malformed or interrupted one none |
| `record`, `message` | the trial's `summary.json`, with every measurement, and why a trial without values ended as it did |

No trial is ever told a value no evaluation produced.

### Decide in `propose`, remember in `proposed` and `tell`

A strategy is told every proposal the ledger kept (`proposed`, with the values as the ledger holds
them) and how each trial ended (`tell`), in the order the ledger kept them: with several
[workers](#trials-at-once---workers), trials end in another order than their numbers. Whenever a
study opens, a resume included, the strategy is made anew and told every proposal and ending its
ledger holds, in the order they were written: the very calls a strategy that ran all along got.
`propose` is never called for a trial the ledger holds. So a strategy that decides in `propose`,
and changes its state only in `proposed` and `tell`, goes on after an interruption exactly where it
was.

What it decides that it could not decide again from those calls (a random draw, a model's answer,
a whole batch it asked for at once) belongs in the note of its proposal, a JSON object the ledger
keeps with the trial and gives back to `proposed`. A strategy that draws at random can also derive
a seed from its settings and the trial's number. The values and the note together may take 64 KiB
of JSON; a transcript belongs in a file of the strategy's own, which the note names by its SHA-256.
k4opt cannot check this rule: a strategy that decides in `tell`, or that asks a remote model again,
proposes otherwise after a resume than an uninterrupted run would have, while the ledger and the
records stay right. k4opt cannot make a remote model answer the same twice: what a study can
reproduce is its recorded proposals.

To propose a batch, propose several trials and answer `WAITING` until enough of them ended; k4opt
asks only when a slot is idle, so a batch larger than `--workers` waits for slots. `propose` runs
in k4opt's own process: a slow one (a model's answer) leaves slots idle meanwhile.

A proposal of values that are no candidate of the study (a parameter it does not search, one
missing, a value of another kind or outside its domain) is a `malformed` trial: nothing is
evaluated, the ledger keeps the values as given, the strategy is told it proposed them, then that
the trial is malformed, and the study goes on. A proposal the ledger cannot keep as given is no
trial: not a `Proposal` (nor `WAITING` or `EXHAUSTED`), values or a note that are no plain JSON
object (NaN, an object), more than 64 KiB of them, or something that looks like a credential. It
stops the study, as an exception of the strategy does; trials under way end as interrupted, or
kept with their record, and `--resume` goes on.

### Settings, files and credentials

What a strategy is given has three roles, which a study records differently:

| | Example | Recorded |
| --- | --- | --- |
| settings | a model's name, a temperature, a batch size, a step, a seed | as they are, under `strategy.settings` in `study.json`, published with the study |
| files | a prompt template the settings name | by bytes and SHA-256 under `strategy.files`; the class names those settings in a `files` attribute (`files = ("prompt",)`) and is given each as an absolute path |
| credentials | an API key | nowhere |

`--strategy-settings FILE` is a YAML mapping, read strictly (a key given twice is refused), and a
file a setting names is relative to it; from Python, `strategy_settings` is a JSON object, and a
relative file is relative to the current directory. A file setting is checked again before each
proposal is kept, as the strategy's code is: one that changed stops the study.

A credential belongs where the strategy runs, not in its settings: read it from an environment
variable, or from a file the strategy opens itself. Settings that look like they hold one (an
`api_key: sk-...`, a token) are refused before anything is created, by the rule
[`k4opt publish`](publication.md) applies, since a study whose `study.json` holds one could never
be published; so is a note that looks like it holds one. An environment variable named as a
credential (`ANTHROPIC_API_KEY`, `HF_TOKEN`) is no part of [what a study runs
on](software.md#the-variables), so a key rotated between two sessions does not refuse a resume.

### What a study records, and resuming it

`study.json` records a strategy of one's own as

```json
{"name": "custom", "class": "CompassSearch",
 "source": {"bytes": 5657, "sha256": "c7a2db40...", "helpers": {}, "executed": true, "revision": null},
 "settings": {"step": 0.25, "min_step": 0.02}, "files": {}}
```

its class, the source it was loaded from as an evaluator's is identified (the file's bytes and the
local modules it imports, and the commit of its checkout), its settings and its files, without
where any of them is; the [ledger](#how-a-study-is-kept-the-ledger-and-its-strategy) commits to
it. The study keeps a private copy of the code and the files, with where they were, in its
[`software/`](software.md) directory. A resume is given the same strategy and settings again
(`--strategy FILE.py:Class --strategy-settings FILE`): k4opt never runs code because a study file
names it. It is refused, saying what differs, for another class, another file or module it
imports, other settings or another file's content; the strategy's checkout at another commit only
warns. `k4opt study`, `load_study`, the JSON form and the dashboard need none of it.

A strategy of one's own runs a problem without inner levels for now; choosing a strategy per level
(yours for the geometry, Optuna's below it, say) is still to come.

## What a study will do: `k4opt problem explain`

`k4opt problem explain` prints what a new study does, as the study itself does before it starts,
with the same `--trials` and `--fix`. It reads the problem file alone, so it needs neither evaluators
nor Key4hep: nothing runs, nothing is written, and the geometry is not read, so a geometry target
that does not resolve is noticed only when the study starts. The levels of a
[nested study](#nested-studies) read as loops, each with its budget and what picks its best trial,
and the last line says at most how many times each stage and each evaluation runs, over all samples
and levels:

```bash
k4opt problem explain examples/nested/cld-pi0-nested.yaml --trials 8
```

```text
For each ECAL cell size (8 trials):
  tune Pandora ECAL hit threshold: 12 trials maximizing π⁰ photons resolved
  evaluate the best again on new events (seed 42 → 506952155)
compare them by π⁰ photons resolved (maximize) and Readout channels (minimize)
at most, if every stage succeeds: simulation 16 · reconstruction 104 · pfo.pi0 104 · readout 8
```

An evaluation runs for every trial. A stage runs only for what it is made from, since trials reuse it
otherwise: once for each trial down to the deepest level whose parameters reach it, on the events of
its level, and each study of a level proposes values of its own. Values repeated, a stage reused from
an earlier study and trials that end early all lower the count. A stage that fails raises it: the
next trial that needs it tries it again, so a simulation that always fails runs in every trial that
shares it. The count is no time either: each candidate that sets an argument of a reconstruction is
also checked by a `k4run --dry-run` first, about half a second each, one after another.
A parameter of an inner level that reaches the simulation gets a warning: each trial of that level
then simulates anew, which is rarely what a level below the geometry is meant for.

## Trials and their outcomes

| The trial | Status | The strategy is told | Values | Constraint values |
| --- | --- | --- | --- | --- |
| a `valid` record, every limit satisfied | `valid`, feasible | `valid` | the objectives | all zero or below |
| a `valid` record, a limit violated | `valid`, infeasible | `valid` | the objectives | at least one above zero |
| an `invalid` record | `invalid` | `invalid` | none | none |
| a `failed` record | `failed` | `failed` | none | none |
| a required evaluation's `check` rejected the candidate, or a required reconstruction's job does not take the arguments it sets; no record, no directory | `rejected` | `rejected` | none | none |
| the strategy's proposal is no candidate of the study: values of other parameters, a value missing, of another type or outside its domain; nothing evaluated | `malformed` | `malformed` | none | none |
| ended without a record | `interrupted` | `interrupted` | none | none |
| the trial stopped the study | `error` | (the study stops) | none | none |

An invalid, failed, rejected or malformed trial never gets made-up values (0, infinity) or
constraint values. TPE takes every trial without values as a failure, which it ignores, so a
region that is always invalid or rejected can be proposed again. Treating invalidity as a
constraint would be a scientific choice, and it is not made.

A rejected trial has no record: nothing of its candidate was created. What the study keeps of it
is in its ledger: the required evaluation whose `check` rejected it (`gamma_10gev/calo.photon`),
or the required reconstruction whose job refused an argument a parameter sets
(`pi0_50gev/reconstruction`), and why; a study read back holds exactly that or refuses the
trial: a rejection by a diagnostic, or with a trial directory, is not one k4opt made.

A **malformed** proposal is a fault of the strategy, not a finding about a candidate: k4opt
checks every proposal before anything of it is made, keeps it as a trial with the reason
(`the strategy proposed a value outside the study's domain: ecal_cell_size: 40.0 is outside the
bounds [2.5, 20.0]`), tells the strategy, and goes on. It counts among `--trials`, has no
directory, and its candidate holds only the values the study fixes. Neither built-in strategy
proposes one; a strategy of another kind can.

The record's status is decided by its required evaluations: those an objective or constraint
reads, those marked `required: true`, and what they read
([problem](problem.md#evaluators)). A diagnostic that fails, is invalid or is blocked leaves a
`valid` trial with its values; the study lists the diagnostics when it starts, the
trial's line ends with `(1 diagnostic did not succeed)`, and `study trial` shows each one.

**The study goes on** after a `rejected`, `invalid` or `failed` candidate, also after many in a
row: a failure can be a legitimate finding (ddsim refusing an overlapping geometry, say). So does
a stage or an evaluation stopped at its time limit, and a geometry whose worker died or ran out
of time; without a `timeout`, a candidate on which Geant4 stalls, or an evaluator that never
returns, holds up an unattended study until it is stopped. Each trial is printed as it ends, so
a broken environment shows at once.

**The trial is an `error`, and the study stops,** when
([the kinds](execution.md#what-ends-a-candidate-and-what-stops-a-session)):

- an evaluator's `check()` raises anything but `CandidateRejected` (`cls.reject`), or anything
  else is wrong before its evaluation starts;
- a file the problem names, a file the geometry's external references read, the environment or
  k4opt's code changed since the study's session opened, also while the trial ran;
- a result breaks the evaluation contract (a `valid` result without a metric an objective or a
  constraint names of it, a wrong unit), or an evaluator raises anything but `EvaluationFailed` or `EvaluationInvalid`;
- the record does not hold up when it is read back (below);
- a value cannot be held by a float;
- the strategy fails to propose or to take a result: this also stops a study outside a trial,
  as an `OptimizationError`, while a bug of k4opt propagates as itself;
- writing the ledger fails: then nothing more is written, and the trial, as any other under way,
  is left running for `--resume` to settle ([below](#interruptions-the-lock-and-recovery)).

Each evaluator's settings and collections are checked once, when the study opens and before
any file is created, so a mistake in them costs no trial. They need no candidate, so an
evaluator's own `check()` is only ever asked about candidates a strategy proposes, and what it
rejects is a finding about that candidate.

**The study keeps, and the strategy is told, what the record holds.** After each evaluation,
`summary.json` is read back from disk and checked:

- it was made for this study and trial;
- its candidate is the trial's, type and value, on the problem's own events (replicate 0, not
  [another replicate](execution.md#k4opt-evaluate) of it);
- its evaluators, [software](software.md), k4opt code, input files and the files the geometry's
  external references read are the study's;
- a file that only candidates name (a `ref` parameter choosing `${CAD}/other.stl`), which the
  study cannot know beforehand, reads as it did for every earlier trial that named it, in this
  session or an earlier one;
- it was made from the candidate alone (no supplied stage);
- its assessment holds up when recomputed from its raw measurements
  ([`Assessment.from_summary`](record.md#assessment-objectives-limits-and-feasibility)).

Only then does the ledger keep the trial with its record, and only then is the strategy told.
This is also how a resume finishes a trial that an interrupted session left with a record, so a
recovered trial is kept exactly as a live one would have been.

## Trials at once: `--workers`

`--workers N` (`optimize(..., workers=N)`) runs the tasks in N worker processes of this machine,
up to N at once: the geometries, simulations and evaluations of one trial and of several, as
[`k4opt evaluate --workers`](execution.md#execution-inline-or-in-worker-processes) runs them.
`k4opt optimize` runs one worker by default, `optimize()` inline unless told otherwise (its
evaluators may be classes of a script or a notebook). A simulation on several threads
(`numberOfThreads`) holds as many slots, and `--workers auto` takes as many workers as the
processors this process may use (on lxplus: the whole node, so give a
number there). N bounds the processors k4opt keeps busy, not the memory: evaluate one candidate
with `k4opt evaluate` first, which prints an estimate of what a task needs while it runs, and
choose N so that the tasks that can run at once fit, with some room, and the study's own process
too: N single-threaded simulations need N times that, simulations on T threads hold T slots each,
so only N/T of them run, each with its memory once. Every trial's record holds the peak memory of
each of its tasks.

```bash
k4opt optimize examples/problem/cld-ecal-granularity.yaml \
  --evaluator examples/evaluators/calo_photon.py:CaloPhoton \
  --trials 32 --seed 42 --workers 8 --output runs/cld-ecal-study
```

- **The strategy is asked for a candidate only when a slot would otherwise be idle**, and only
  after every trial that ended was kept and told. So at most N trials run at once, and fewer when the
  trials under way have tasks enough for every slot. A trial whose simulation waits for the same
  simulation of another holds no slot, but still counts among the N, and so does one that ended
  but is held until the record of the trial whose output it reused is written: the strategy is
  not asked for trials that would only wait too, nor for more trials than N it has no answer of.
  A strategy may also answer that it waits for a trial under way to end before it proposes
  again (one that proposes in batches, say); that holds no slot either.
- **Each trial is told as it ends**, in the order the trials end, after the same checks of its
  record as with one worker; its line is printed then.
- **Every trial keeps its own directory and record**, and a record is the same whatever N
  (only `environment.observed.execution` says how its tasks ran). `--workers` is not part of the
  study: a study may be resumed with any N.
- **A trial that stops the study** ([above](#trials-and-their-outcomes)) stops the trials still
  running too: their simulations and evaluations are killed, their directories kept as they
  are, and each is ended as `interrupted`, in number order.
- **Each trial is kept once and told once.** The ledger holds how a trial ended before the
  strategy hears of it; a strategy that fails as it is told stops the study, with the trial
  kept, and the next session makes the strategy anew from the ledger, never telling it twice.

**Reproducibility.** A grid proposes the same candidates whatever N. With Optuna and one worker
(or inline), a seed proposes the same candidates as ever. With more, what the sampler proposes
depends on which trials have ended when it is asked, so two runs of one seed take different
paths; and the trials still running when the tenth completes were proposed at random too, so up
to N - 1 more than with one worker. Each trial still records its candidate, what its strategy
noted of it (Optuna's session seed, the grid's point) and its record, and the records of a
candidate are the same whatever N: the simulation seed is each sample's own.

**Stages made once.** A trial does not run a stage again whose output a stage with the same
signature made already, in this session or in an earlier trial of the study (a resume reuses the
study's own trials, at every level of a nested one), or in the runs and studies `--reuse DIR`
names (repeatable; a directory is searched for runs at any depth): a study that
varies only the reconstruction, or only evaluators' settings, simulates once. The trial's record
says where the reused output comes from ([reuse](execution.md#stages-made-once-reuse)). Neither
the reuse nor `--reuse` is part of the study: a reused stage is the stage that would have run.

## The study directory

```text
runs/cld-ecal-study/
├── study.json        what the study is made with, written once (k4opt.study/1)
├── trials.jsonl      the ledger: every trial and how it ended, one event per line
├── .lock             while a process works on the study: its host, process and start time
├── trial-000000/     what k4opt evaluate writes: geometry/, simulation/, evaluators/
│   └── summary.json  the record; "run": {"study": <study id>, "trial": 0}
└── trial-000001/ ...
```

Nothing in a trial directory is ever rewritten or reused, and nothing is deleted: every trial
keeps its geometry and its EDM4hep file.

`study.json` holds:

- the study's `id`, creation time and `name` (`<problem name>-<8 hex>`, its whole id for a study
  below the top);
- the problem's path and normalized definition;
- the path, size and SHA-256 of every file the problem names, by where it names it
  (`samples.<sample>.<stage>.file`, `samples.<sample>.simulation.<option>` for a file a ddsim
  option names);
- `implementations`: each evaluator the problem binds, once however many entries bind it, by
  name: its version, the stage it reads, its source identity, declared metrics and declared
  artifacts and roles;
- the compact file, source root and fingerprint of the unmodified geometry, and the path and
  SHA-256 of each file its external references read (`${DD4hepINSTALL}/...`, `${K4GEO}/...`,
  absolute paths);
- `KEY4HEP_STACK` and `K4GEO`;
- `programs`: each program of the stack a stage that runs starts (`ddsim`, `k4run`; not that of
  a stage given as a file), as the executable the session found on `PATH` when it opened, where
  its links lead, with its size and SHA-256, and the `interpreter` its `#!` line names as `PATH`
  found it (`null` for a program that is no script); `null` for a program not found;
- `software`: what it runs on, the build of the stack, the system and the SHA-256 of every
  variable not known to be irrelevant, with their `id` ([software](software.md));
- k4opt's version, the SHA-256 of its code and its git checkout;
- the `identity` digest;
- `scope`: what the study searches (`searched`, the parameters its strategy proposes), the
  values it `fixed` for every trial, the `level` and `replicate` of its trials' records, and the
  trial of another study it was started for (`parent`; [nested studies](#nested-studies));
- `strategy`: what proposes its candidates. For `optuna`, its version, the sampler with its
  seed and settings, and the searched parameters' distributions; for `grid`, the values of each
  parameter it and the levels below it search, and how many points its own grid has.

Combined objectives use `binding/stage/metric` without a sample prefix. Two combined bindings
reporting the same metric have separate objective columns. `study show` and `study trial`
include the role wiring. Changing sample properties, role wiring or declared role requirements
refuses resume, just as changing evaluator settings does. A failed or invalid upstream input
blocks the combined evaluation; when an objective or constraint reads it, the candidate is
`failed` or `invalid`, with no partial objective vector.

## Nested studies

A problem with an [`optimization` block](problem.md#optimization) with `inner` levels runs as a tree
of studies, in one session, with the same `k4opt optimize` (`--trials` is the budget of the top
level). For each trial of a level, a study of the level below searches its parameters with the
trial's values fixed, with the strategy of the top (a grid of the values given for its
parameters, or Optuna seeded from the trial). Once it ran its `trials`, or its strategy is
exhausted, its best feasible trial by its `objective` (the
lowest number among equals) is *selected*, and that candidate is evaluated once more, on new events
(its seeds derived: [replicate](execution.md#k4opt-evaluate) = the levels below), for the level's
own criteria. That record is the trial's: its objectives and limits are judged on those new events,
so a selection that is infeasible there makes the trial infeasible.

```text
runs/nested/
├── study.json, trials.jsonl
├── trial-000002/summary.json      trial 2's record: the selected candidate, on new events
└── inner/
    └── trial-000002/              the study trial 2 stands for, laid out as any study
        ├── study.json             scope: level 1, the values of the level above, parent
        ├── trials.jsonl           its own ledger
        └── trial-000007/summary.json
```

- A study whose level below has **no feasible trial** leaves its trial `unscored`: no record and
  no value, the strategy told so, and its message says why (`no feasible trial among 20: 3
  invalid, 17 infeasible`).
- The trials of all open studies run side by side on the session's workers
  ([`--workers`](#trials-at-once---workers)): a study of the level below is asked first, a new trial
  of the level above only when no open study has a candidate. Their stages are reused as any
  stages are: a geometry the trials of a study share is simulated once. A study of a level below
  whose trials share a stage (one none of the parameters it or a level below searches reaches,
  for any of those levels' evaluations) asks for one trial first, and for the others once that
  ended: they would only wait for the shared stage, taking places among the N trials under way
  that other studies could use, and would be proposed knowing nothing. Meanwhile the workers go
  to the other studies, so the geometries of as many top trials as there are workers are
  simulated at once; a worker nothing else would use takes a further trial of a study held back
  all the same. A resumed study is held back again until one of its new trials ended (what its
  trials share may be gone). This decides when trials are proposed, so it changes what a
  learning strategy proposes, never how a candidate is evaluated.
- **Interrupted** at any point, the run goes on with `--resume`, from what is stored. Before
  anything new runs, it reads the tree as `load_study` does, and refuses one that does not hold
  up. A trial of a level above stays running, its study of the level below is resumed (`--trials 0` adds no new
  trial of the top), its selection is made or checked again, and a selection that no longer is the
  study's best stops the run. What an interrupted evaluation of the new events left in a trial
  directory is moved to `inner/interrupted-trial-NNNNNN-*`. A study of a level below is resumed
  through its top study alone, whose lock covers the tree.
- `load_study` reads the tree: `trial.inner` is the study of the level below, `trial.selected` the
  trial chosen and the values it adds, each link checked (the study was started for this trial,
  with its values and the strategy the trial's study gives it: the grid values of its
  parameters, or the same sampler seeded from the trial; it ran all its trials or its strategy
  was exhausted, the selected trial is its
  best by its records, and the
  trial's candidate is those values with that trial's). The JSON form holds the
  inner studies. `k4opt study show` lists them; `k4opt study show DIR/inner/trial-000002` shows one.

The new events keep a level from being judged by the luck of its own tuning, but the top level
still chooses among its values: the best of them is optimistic in turn. `k4opt evaluate --replicate
N` of the winner (`N` above the number of inner levels) gives an independent check.

## Resuming a study

`k4opt optimize ... --resume --output DIR --trials N` runs `N` more trials of the study in
`DIR`. A study resumes only with what it was made with, compared by content and never by
location. Moving the problem directory, an evaluator file, the k4geo tree or the study
directory itself is fine: a trial's record names the files of its run directory relative to it
([Artifacts and the files of a run](evaluation.md#artifacts-and-the-files-of-a-run)).

| Part | Compared: must be equal | Recorded only |
| --- | --- | --- |
| problem | its definition, but not its `name` | the path of the file |
| files the problem names | size and SHA-256 of each | their paths |
| evaluators | each one the problem binds: name, version, SHA-256 of the source file (one without a hash, or whose hash is not known to be of the code that runs, `executed: false`, cannot make a study) and of the modules it imports from beside it (`helpers`), the metrics it declares with their units and the artifacts it declares (read from its class, where an evaluator declares them) | path, the reference it was loaded by, whether the bytes were `executed`, the git revision (a change warns) |
| geometry | the fingerprint of the unmodified geometry, which covers every reachable file below its root, the attribute each geometry target of the parameters resolves to (its file below the root, locator, attribute and unmodified value; every trial's record must change exactly those) and the SHA-256 of each file its external references read (`null` for one that names no file) | compact path, source root, the paths the external references resolve to, `K4GEO` |
| software | its `id` (the build of the stack, the system and every variable not known to be irrelevant: [software](software.md)), its build and its system; every trial's record must have run on the same | `KEY4HEP_STACK` |
| programs | each program the stages run (`ddsim`, `k4run`): the path of its executable, where its links lead (a release of the stack is told apart by its path), its size and SHA-256, and its interpreter; every trial's record must have run the same | |
| k4opt | the SHA-256 of its code | version, git commit |
| strategy | its name; for Optuna, its exact version, the sampler and its settings and seed, and the distributions; for the grid, its values and points; and that it is the one the study's ledger committed to when the study was made | |

The `identity` digest covers the compared column only, but for the programs, which are compared
apart, so that a refusal names the program that differs. Both
ddsim and k4run are Python scripts: the same script with another `PYTHONPATH` (a local build
layered over the release) runs other code, so any difference in the software refuses a resume,
naming the variable or the build. **Resume a study the way it was started**: a login shell and a
bare service unit differ in dozens of variables, and each refusal names them. A resume is refused,
with what changed, for any difference in it. It is also refused for `--strategy`, `--grid` or
`--seed` with `--resume` (a study keeps its strategy, grid and seed; one of its own is given again,
with its settings, and compared), for a recorded strategy (sampler, seed, startup trials,
distributions; a grid's values; the class, source, settings and files of one of its own) other
than the one this k4opt would run the study with, and for a directory that holds no `study.json`; using a study
directory without `--resume` is refused too. A changed Optuna version (for a study of its
strategy) and changed k4opt code have no override: install the recorded Optuna version, and run
a long campaign from a k4opt checkout of its own.
A resume only warns when an evaluator's git checkout is at another commit or has other
uncommitted changes: the evaluator's file and the modules it imports are unchanged (a change of
either refuses the resume), but other files beside them, such as data it reads, may not be.

Within a session, the files the problem names, the files the geometry's external references
read, `KEY4HEP_STACK`, `K4GEO`, the software, the programs and k4opt's code are checked before each
candidate and again after it ran, before its record is written; the geometry's own files are checked by its
materialization. A change while a candidate runs leaves its directory without a
`summary.json`, and the trial stops the study.

**Seeds.** With Optuna, each session samples with a seed derived from the study's seed and the
number of trials already in the study. Otherwise a resumed session would replay the first
session's random proposals (with Optuna, the same seed repeats them). What this does not
guarantee:

- TPE can still propose an earlier candidate, which is likely for integer and categorical
  parameters. It is evaluated again as a new trial with a record of its own, which spends
  budget but corrupts nothing, since the simulation seed repeats its result;
- an interrupted study does not follow the path an uninterrupted one would have taken.

Each trial records the seed of its session, and the same history, seed and Optuna version give
the same proposals again. A grid needs no seed: it goes on from its next point.

## Interruptions, the lock and recovery

**Ctrl-C and SIGTERM** (`systemctl --user stop`, a batch system ending a job) stop the study
cleanly. Each running trial is `interrupted`, or is kept with its record if one was published
just before; ddsim is killed, with any process it started, and so are the worker processes; the
lock is removed; the command exits 1 and says how to resume. A study killed outright (SIGKILL)
takes its worker processes with it (Linux), and a resume ends the trials it left running.
Wherever an interruption falls, even within a write to the ledger, the study stays readable and
resumable: each trial reads as what it ended as, or as `running` until a resume ends it.

**The lock.** `.lock` is created exclusively when a process starts working on a study, and
removed when it ends, so two processes never work on one study. A lock that a killed process
(SIGKILL, a crashed node) left behind is never taken over. The error names its host, process
and start time and, on the same host, whether that process still runs. Once nothing works on
the study any more, remove the file named and resume.

**Recovery.** Under the lock, before anything is written, a resume reads every trial of the
study as `load_study` reads it, and refuses a study that does not hold up: a record that is
gone or was changed (other bytes than its ledger names), one not made for its trial, a ledger
that does not hold together (below), or a `trial-*` directory that no trial names. Then it
begins its session in the ledger and ends the trials a session left running:

- a trial without a directory, or whose directory has no `summary.json`, is `interrupted`; its
  directory is left as it is and never rerun or reused;
- a trial with a `summary.json` is kept with that record, after the same checks as a live one;
  a record that does not hold up makes the resume refuse.

A study whose setup a killed process cut short, before its ledger was made, is set up by the
resume. Recovered trials are printed first, marked `(recovered)`, but they do not count among
the `--trials` of the resume, and interrupted trials are not retried automatically (a grid's
interrupted point is spent). Only then is the strategy made, from every trial as it now stands.

## How a study is kept: the ledger and its strategy

A study owns its trials; its strategy only proposes.

| | Owned by | Kept |
| --- | --- | --- |
| what a study is made with, its scope, its strategy's settings | the study | `study.json`, written once |
| which trials there are, what was proposed for each, how each ended, which trial of a level below each selected | the study | its ledger, `trials.jsonl` |
| what each trial measured: candidate, assessment, provenance, diagnostics | the evaluation | `trial-NNNNNN/summary.json`, named in the ledger by id and SHA-256 |
| what a strategy learnt (Optuna's model of the trials) | the strategy | nothing on disk: made anew from the ledger whenever the study opens |
| the outputs a trial's stages made, and where reused ones come from | the evaluation | the trial's directory and record |

**The ledger** is `trials.jsonl` in the study directory, one JSON object per line, each an
*event* numbered by `seq`: `study` (the first line, naming the study and committing to the
strategy it was made with by the SHA-256 of what `study.json` records under `strategy`; a study
whose `study.json` records another one is refused, read back or resumed), `session` (a session
opened the study), `proposed` (trial `N` with the values its strategy proposed, checked, and
what the strategy noted of them), `malformed` (a proposal that is no candidate, with the values
it gave, as given, and why),
`selected` (the trial of the study below that a trial selected), one ending of a trial
(`recorded` with the status, id and SHA-256 of its record; `rejected`, by what and why;
`unscored`; `interrupted`; `error`, why) and `exhausted` (the strategy has nothing more to
propose, kept as soon as it says so: right after its last proposal, or when asked for one). Only the process that holds the lock writes it, only by appending a whole line, synced
to disk before anything else happens, and every event is checked as a reader checks it before it
is written. A line is written once it ends in a newline: a line a killed process cut short is
the last one and was never written, so a reader leaves it out and the next session removes it.
A write that fails may or may not have reached the disk, so nothing more is written after it:
the next session settles what was being written from what is on disk. A trial a later session
ends is `recovered`. Trials are numbered from 0, in order, and the `seq` of a trial's proposal
and of its ending say in which order trials were proposed and ended, which their numbers do not
when several run at once; a trial is proposed once and ends once; a ledger that holds anything else (a line that is not the next event, a trial ended
twice, a gap, another study's first line) is refused, by `load_study` and by a resume alike.

**The order of writes.** A trial goes through these steps, each of which an interruption can be
the last:

1. the strategy proposes (nothing is kept);
2. the ledger keeps the proposal, checked: the trial is running; the strategy is told it was kept;
3. the candidate is evaluated into its directory;
4. its `summary.json` is published, whole;
5. the record is read back and checked, and the ledger keeps the trial with it;
6. the strategy is told how it ended.

Stopped after 1, the next session proposes again; after 2 or 3, the trial is `interrupted`;
after 4, the next session keeps it with its record as a live one would have been kept; after 5
or 6, it is kept, and the next session's strategy learns it from the ledger. No trial is lost,
kept twice, or told a value its record does not hold, and no strategy is told what the ledger
does not hold: a strategy keeps nothing of its own to fall out of step with the ledger. A
strategy that fails as it is told (6) stops the study with the trial kept; a nested trial's
selection is kept before its selected candidate is evaluated again, and checked again on every
resume.

**The strategy boundary.** A strategy is asked for a proposal for trial `N` when the study has a
slot for it and answers with values for the parameters the study searches (and a small note,
recorded with the trial), with *waiting* (nothing until a trial under way ended) or with
*exhausted*, and may say after each proposal whether it is exhausted. It is told each proposal
the ledger kept (*proposed*: the values as the ledger holds them, also those of a malformed one)
and how each trial ended (*tell*): the trial's status, and for a valid one its objective values,
the normalized margin of each limit and its feasibility, all from the checked record, the trial
as `load_study` reads it back, and where the record is. When a study opens, the strategy is made
anew and told every proposal and ending the ledger holds, in the order they were written (by
`seq`): the very calls a strategy that ran all along got, also when trials ended in another order
than their numbers. A proposal the ledger cannot keep as given (no proposal at all, values or a
note that are no plain JSON object, more than 64 KiB of them, or something that looks like a
credential) is no trial: it stops the study, and `--resume` goes on. A strategy executes nothing,
changes no objective or limit, and keeps no trial; k4opt checks every proposal against the
parameters' domains before anything is made of it.

## Inspecting and exporting a study

`k4opt study` reads a study back as its ledger and records hold it. Nothing is evaluated,
created, locked, recovered or written, so it can read a study while a session works on it, a
copy of the study directory, or a study long after it ran: the problem file, the evaluators, the
geometry, the Key4hep stack and the strategy it was made with (Optuna, too) need not exist any
more. What the study optimizes comes from `study.json`, which trials there are and how each
ended from its ledger, and every value from the trials' `summary.json`.

```bash
k4opt study show runs/cld-ecal-study
k4opt study trial runs/cld-ecal-study 12
k4opt study export runs/cld-ecal-study --format csv --output results.csv
k4opt study export runs/cld-ecal-study --format json --output results.json
```

`show` prints what the study optimizes, how its trials ended, and the trials on the Pareto front
with their values to 6 significant digits (the values here are illustrative). It names no single
"best" trial: with several objectives, the front is the answer.

```text
study:      cld-ecal-granularity-3f9c2a1b, created 2026-10-30T12:00:00+00:00
id:         3f9c2a1b-5d0e-4f43-9a52-77c1d2e0b6a4
directory:  /home/me/runs/cld-ecal-study
strategy:   optuna 5.0.0, TPESampler, seed 42
parameters: ecal_cell_size  continuous [2.5, 20.0] mm  (geometry constants.ECal_cell_size)
objectives: gamma_10gev/calo.photon/simulation/ecal_barrel.angular_residual  minimize  (mrad)
            gamma_10gev/calo.photon/simulation/ecal_barrel.fired_cells  minimize  (1)
limits:     none
trials:     30: 28 valid (28 feasible, 0 infeasible), 1 invalid, 1 failed
pareto:     0, 4, 7
            trial 0:  ecal_cell_size=9.79789 mm  gamma_10gev/calo.photon/simulation/ecal_barrel.angular_residual=0.948 mrad  gamma_10gev/calo.photon/simulation/ecal_barrel.fired_cells=227 1
            ...
```

`trial` prints one trial, chosen by its number, with every value as recorded: its status, its
candidate with units, each objective with its uncertainty and unit, each limit with the value it
judges, its bound, its margin and whether it is satisfied, the wall-clock time of the whole
evaluation and of each stage that ran, its record (path, id, SHA-256), the message of an
invalid, failed or error trial, and each diagnostic that did not succeed, with its status and
message. Every text from the study (names, units, paths, ids, messages)
is shown with its control characters escaped (`\x1b`), so neither a colored log in a message
nor anything else a study holds can change the terminal. Times are those the record
holds; nothing is estimated.

| The trial | Status | What is shown |
| --- | --- | --- |
| a valid record, every limit satisfied | `valid`, feasible | objectives, limits with margins |
| a valid record, a limit violated | `valid`, infeasible | the same; never on the front |
| an invalid or failed record | `invalid`, `failed` | no objective values; limits not assessed; its message |
| rejected by an evaluator's `check` | `rejected` | nothing measured, no record; the evaluation that rejected it and why |
| a proposal that is no candidate of the study | `malformed` | nothing measured, no record; why, and the values the study fixes |
| ended without a record | `interrupted` | nothing measured |
| stopped the study | `error` | nothing measured; why it stopped. A record it may have left is not read |
| running in the ledger | `running` | nothing measured or checked; a record already on disk is not read: the study has not kept it yet, a resume will |

A running trial is being evaluated, or was left by a session that was killed; the resume closes
it. Counts cover every trial of the study, from all its sessions, recovered ones included.

**What is checked.** `study.json` must be `k4opt.study/1`, and its identity digest that of its
parts, so an edited problem definition is refused. Its ledger must be the study's and hold
together ([the ledger](#how-a-study-is-kept-the-ledger-and-its-strategy)): every trial proposed
once, in order, with values of the parameters the study searches, each in its domain, ended once.
Every trial the ledger keeps with a record must still have it, with the bytes whose SHA-256 the
ledger names and the id it names, made for this trial of this study with its candidate,
evaluators, software, k4opt code and files, with the status the ledger holds, and its assessment
must recompute from its raw measurements
([`Assessment.from_summary`](record.md#assessment-objectives-limits-and-feasibility)). A
rejection must be by a task that can reject, and nothing of a rejected or malformed candidate may
exist. A record that is missing or does not hold up, a ledger that does not hold together, or a
study that was never set up makes the read fail, with the trial and file named: nothing is
skipped, repaired or recomputed with other code. A read checks the records, not the files they
name: it never opens a trial's events, artifacts or geometry, so it says that each record is the
one the ledger names and holds up, not that those files still exist or still hold what the
record says they held. A `trial-*` directory that no trial names is a warning. Warnings go to
stderr.

**The Pareto front** holds the feasible valid trials that no other one dominates: at least as
good in every objective, in its direction, and better in one. Trials with equal values all stay.
The recorded values are compared and their uncertainties play no part, so the front is the one
Optuna would report for the trials, whatever proposed them.

**The ledger is only read.** A line a session is writing as the study is read, not yet whole,
is not read; nothing is locked or written.

### Export formats

`export` writes the data alone: to standard output, or to `--output FILE`. The file must be new
(an existing one is never replaced) and outside the study directory (refused there, also through
a symlink); it is written whole, so a failure leaves no partial file. The same study gives the
same bytes every time. Values are as recorded: a missing value is empty in CSV and `null` in
JSON, never 0 or NaN, and a float keeps all its digits.

**CSV**: one row per trial, by number, with columns that only the problem decides:

| Column | Holds |
| --- | --- |
| `study` | the study's id |
| `trial` | the trial's number |
| `status` | `valid`, `invalid`, `failed`, `rejected`, `malformed`, `unscored`, `interrupted`, `error` or `running` |
| `feasible` | `true` or `false` for a valid trial, empty otherwise |
| `pareto`, `recovered` | `true` for a trial on the front, for one a resume finished |
| `parameter:<reference>` | the candidate's value (its unit is the problem's) |
| `objective:<sample>/<evaluator>/<stage>/<metric>`, `…:unit`, `…:uncertainty` | the measured value, its unit, its uncertainty (without `<sample>/` for an evaluator of the candidate) |
| `limit:<sample>/<evaluator>/<stage>/<metric>:<bound>:value`, `…:margin`, `…:normalized`, `…:satisfied` | the value judged, its margin (zero or below when satisfied), the margin in units of the constraint's scale (what the strategy was told), the verdict |
| `elapsed_seconds`, `stage:<sample>/<stage>:elapsed_seconds` | the recorded wall-clock time of the evaluation, and of each stage of each sample that ran (`stage:gamma_10gev/simulation:elapsed_seconds`) |
| `record`, `record_sha256`, `record_path` | the record's id, SHA-256 and path in the study |
| `message` | why the trial is invalid, failed or an error |
| `diagnostics` | how many of its diagnostics did not succeed; empty without a record |

In a column name, a `%`, `/`, `:` or control character within a name from the problem (a
parameter reference, a stage or metric name) is percent-escaped (`%25`, `%2F`, `%3A`, `%1B`), so
each column names one thing and the header holds nothing a terminal would run:
`objective:e/analysis/count%3Aunit` is the value of the metric `count:unit` of the evaluator
`e`, and `objective:e/analysis/count:unit` the unit of `count`. Split such a name at its `/`
and unescape each part (`urllib.parse.unquote`) to get the sample (when there are four parts),
evaluator, stage and metric back; the last two are always stage and metric.
Booleans are `true` and `false`; control characters in text are escaped (`\x1b`), tab and
newline kept in quoted cells. The CSV cannot tell a categorical `""` or `"1"` from an empty cell
or the number 1: the JSON export is the typed one.

**JSON** (`k4opt.study-results/1`): the whole study, typed, and without any location, so it can
leave the machine: what a [publication](publication.md) holds is exactly these bytes. `study` is
`study.json` without its locations: the problem file's path, the paths of the files it names, of
the evaluators' files and of the files the geometry's external references resolve to, the
geometry's source root, `K4GEO`, the paths of the programs of the stack and of their interpreters
(each kept by size and SHA-256) are left out, and so is anything else not on the list in
`k4opt/optimization/_study.py` (`PORTABLE`). Locations are no part of a study's identity, so its
`identity` digest still holds. The study directory and the read warnings are not part of it
either (warnings go to stderr).

```json
{
  "format": "k4opt.study-results/1",
  "study": {"schema": "k4opt.study/1", "id": "3f9c2a1b-...", "name": "cld-ecal-granularity-3f9c2a1b", "problem": {"definition": {}}, "strategy": {}},
  "outcomes": {"feasible": 28, "infeasible": 0, "invalid": 1, "failed": 1, "rejected": 0, "malformed": 0, "unscored": 0, "interrupted": 0, "error": 0, "running": 0},
  "exhausted": false,
  "pareto": [0, 4, 7],
  "trials": [
    {
      "number": 0,
      "status": "valid",
      "outcome": "feasible",
      "recovered": false,
      "candidate": {"ecal_cell_size": 9.797885082295046},
      "assessment": {"feasible": true, "objectives": [], "limits": []},
      "elapsed_seconds": 291.3,
      "stage_seconds": {"gamma_10gev/generation": 0.4, "gamma_10gev/simulation": 280.2},
      "message": null,
      "diagnostics": [],
      "record": {"id": "7a2aca36-...", "sha256": "a22d51..."},
      "pareto": true
    }
  ]
}
```

A trial's `assessment` is the one its `summary.json` holds (`null` without a record): each
objective as `stage`, `metric`, `value`, `unit` and `uncertainty`, and each limit as `stage`,
`metric`, `bound`, `limit`, `unit`, `value`, `uncertainty`, `margin`, `scale`, `normalized` and
`satisfied`. Its
`diagnostics` are those of the record: each diagnostic that did not succeed, as `task`, `status`
and `message`. `record` identifies the trial's `summary.json` by its id and SHA-256; the record
itself stays in the study directory, at `trial-NNNNNN/summary.json` (`NNNNNN` the trial's
number). A trial without a record holds the values the study fixes and those its strategy
proposed (none of a malformed proposal), and a trial with a record every parameter. `exhausted`
says whether the strategy had nothing more to propose. Messages are as the records and the
ledger hold them: an evaluator that writes a path into its message puts it into the export.

`StudyResults.from_dict` reads the JSON form back, without Optuna:

```python
import json
from pathlib import Path

from k4opt.optimization import StudyResults

study = StudyResults.from_dict(json.loads(Path("results.json").read_text()))
```

It checks the form, not the measurements, whose records it does not hold: the identity digest of
`study` (so its problem definition is the study's), each trial's status, outcome, candidate
(each value in its parameter's domain, as a live evaluation checks it), assessment and record
against the problem and each other, every limit judged again on the value it records (its
margin, normalized margin and verdict must be exactly those, and the trial's feasibility theirs),
and the outcome counts and the Pareto front, recomputed. What does not hold up
raises `OptimizationError`. `load_study` checks each record in full before any of it is exported.

## Python API

```python
from k4opt.evaluation import load_evaluator
from k4opt.optimization import optimize

report = optimize(
    "examples/problem/cld-ecal-granularity.yaml",
    [load_evaluator("examples/evaluators/calo_photon.py:CaloPhoton")],  # the class
    "runs/cld-ecal-study",
    trials=30,
    seed=42,  # Optuna's; or strategy="grid", grid={"ecal_cell_size": [2.5, 5.0, 10.0]}
    # or strategy=load_strategy("FILE.py:Class"), strategy_settings={...}: your own
    workers=4,  # worker processes, tasks at once; "inline" by default
    warn=print,  # each warning
    progress=print,  # each TrialResult as its trial ends, the recovered ones of a resume first
)
report.pareto  # the trial numbers on the Pareto front of the feasible valid trials
report.trials  # each TrialResult, as load_study reads it back (below)
```

`optimize` returns the study as [`load_study`](#inspecting-and-exporting-a-study) would read it once its
trials ended, and gives `progress` each trial as that reads it: one interpretation of a trial,
live and later. `resume=True` continues a study with the strategy it was made with (given again,
with its settings, for [one of your own](#a-strategy-of-your-own)).

`optimize` raises `OptimizationError` (a `K4OptError`) when a study cannot start, resume or go
on, and `KeyboardInterrupt` after a clean stop on Ctrl-C or SIGTERM. Underneath it is an
[`EvaluationSession`](execution.md#evaluating-candidates-evaluationsession), which anything that
proposes candidates can drive the same way.

`load_study` reads a study back, as `k4opt study` does:

```python
from k4opt.optimization import load_study

study = load_study("runs/cld-ecal-study")
study.name, study.id, study.created, study.directory
study.problem  # the ProblemSpec of study.json: parameters (bounds or choices, unit),
# objectives (direction), constraints
study.outcomes  # how many trials ended how: feasible, infeasible, invalid, failed, ...
study.pareto  # the numbers of the feasible valid trials on the Pareto front

trial = study.trial(12)  # by its number
trial.status, trial.outcome, trial.candidate, trial.message, trial.recovered
for m in trial.assessment.objectives:  # None for a trial without a valid result
    print(m.stage, m.metric, m.value, m.unit, m.uncertainty)
trial.assessment.limits  # each Limit: value, limit, margin, scale, normalized, satisfied
trial.elapsed_seconds, trial.stage_seconds  # recorded times, stages by label
trial.record, trial.sha256, study.record_path(trial)  # the summary.json it was read from

data = study.to_dict()  # the JSON form, plain data without locations; study.to_json() as text
StudyResults.from_dict(data)  # read back, checked (above)
```

`StudyResults` and `TrialResult` are a snapshot: immutable, detached from the study directory,
and free of any strategy's types; `study.exhausted` says whether its strategy had nothing more to
propose; `study.document` is `study.json` as a read-only mapping (without its locations
for a study read from its JSON form, whose `directory` is `None`), and `study.scope` what it
searches, the values it fixes and the criteria its trials are judged by. `optimize` returns the
same. `load_study` raises `OptimizationError` for anything it cannot read or does not trust
(above), and needs no strategy: Optuna is needed only to run or resume a study of its strategy.

### Trade-offs against a reference trial: `trade_offs`

Of the trials on a front, which to choose is a matter of what one gives for what. `trade_offs`
asks the feasible trials of a study which meet a rule for each objective against a reference
trial:

```python
from k4opt.optimization import Rule, trade_offs

found = trade_offs(
    study,
    12,  # the reference trial: one with objective values
    [
        Rule("gamma_10gev/calo.photon/simulation/ecal_barrel.angular_residual", "worsen", 0.05),
        Rule("gamma_10gev/calo.photon/simulation/ecal_barrel.fired_cells", "improve", 0.10),
    ],
)
found.matches  # the feasible trials (but trial 12) at most 5 % worse and at least 10 % better
found.change[37]  # each objective of trial 37 less the reference's, in its unit
found.excludes  # how many trials each rule excludes: which is the hardest to meet
```

| Rule | Asks |
| --- | --- |
| `Rule(label, "improve", amount)` | better than the reference by at least `amount` |
| `Rule(label, "worsen", amount)` | worse by at most `amount` |
| `Rule(label, "keep")` | not worse |
| no rule | anything |

Better and worse follow each objective's direction. An amount is a share of the reference's
value (`0.05`: 5 %), or with `relative=False` in the objective's unit, which a reference value
of zero needs. By default a difference within the uncertainty of the two values (each trial's,
added in quadrature) counts as none: a trial must improve beyond its uncertainty to improve at
all, and worsening within it is no worsening, so that a trade-off is never decided by noise;
`within_uncertainty=False` compares the values as they are. It refuses (`OptimizationError`) a
reference without objective values, an objective the study does not have, two rules on one, and
a negative amount. It decides nothing of the study: it is a question asked of its results, which
the [dashboard](dashboard.md) asks in its Trade-offs tab.

## Not done

Retrying interrupted trials, reusing the record of a candidate proposed twice, a warm start from
`k4opt evaluate` records, other Optuna samplers and sampler options (those that keep state of
their own, CMA-ES and NSGA-II, need a checkpoint of it in the ledger), a strategy per level (a
strategy of one's own runs a problem without inner levels), a checkpoint of a strategy's own
state, letting the trials under way finish when a strategy fails, a typed view of a record's
other measurements for a strategy, grids of evenly spaced points or of physical units, a characteristic `scale` per constraint, treating invalid candidates as
infeasible, trials on other machines (a batch system), overrides for changed software, comparing
studies, exporting an evaluator's other measurements (each row names its record, which holds
them), and a memory limit of k4opt's own. None of them is blocked by the design: records have
ids, the ledger names them, studies reference them, and neither the session nor the ledger
knows a strategy.
