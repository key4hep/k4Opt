# Optimization and `k4opt optimize`

`k4opt optimize` runs a **study** of a problem. An optimizer proposes candidates. Each one is
evaluated exactly as [`k4opt evaluate`](evaluators.md#k4opt-evaluate) evaluates one, into a
trial directory of the study. The optimizer then learns each result from that trial's
`summary.json`, read back from disk and checked. The optimizer is
[Optuna](https://optuna.org)'s TPE sampler, but k4opt is not built around it. The problem file,
the evaluators and the run records know nothing of Optuna, and only one module of this
component imports it. A study keeps everything it was made with in `study.json`, and it
resumes only with exactly that.

```bash
k4opt optimize examples/problem/cld-ecal-granularity.yaml \
  --evaluator examples/evaluators/calo_photon.py:CaloPhoton \
  --trials 30 --seed 42 --output runs/cld-ecal-study
```

It prints each trial as it ends, and the study at the end (the values here are illustrative):

```text
trial 0:   valid    feasible    constants.ECal_cell_size=9.79789  simulation/ecal_barrel.angular_residual=0.948  simulation/ecal_barrel.fired_cells=227
trial 1:   failed   calo.photon failed: ddsim exited with status 1 (...)  trial-000001/summary.json
...
study:    cld-ecal-granularity-3f9c2a1b, 2 objective(s), 0 limit(s), runs/cld-ecal-study/study.json
trials:   30: 28 valid (28 feasible, 0 infeasible), 1 invalid, 1 failed
pareto:   0, 4, 7, 12, 19
```

| Option | Meaning |
| --- | --- |
| `--evaluator FILE.py:Class` | an evaluator of the problem, as for `k4opt evaluate` (repeatable) |
| `--trials N` | how many *new* trials this invocation runs; each counts, whatever its outcome |
| `--output DIR` | the study directory: new or empty for a new study, the study's own with `--resume` |
| `--seed S` | the sampler's seed, in [0, 2**32); one is drawn and recorded without it. It is not the simulation's seed, which stays `stages.simulation.seed` |
| `--resume` | continue the study in `DIR` (see [Resuming](#resuming-a-study)) |
| `--storage URL` | an Optuna database URL (SQLite on a local disk, PostgreSQL) in place of the journal file in `DIR`; it needs SQLAlchemy, and the same URL on every resume |

The exit status is 0 once every trial asked for has ended, even if candidates were `invalid` or
`failed`. It is 1 if the study cannot start or resume, a trial stops it, or it is interrupted,
and 2 for a usage error. Optuna is not part of the Key4hep stack: `setup.sh` installs the
version pinned in `requirements.txt`, and `pip install 'k4opt[optimize]'` installs it elsewhere.
Everything else in k4opt works without it.

## What the optimizer searches, and what it is told

**Parameters.** Each parameter becomes one distribution, named by its canonical reference:

| Parameter | Searched as | Refused when the study is made |
| --- | --- | --- |
| `continuous [low, high]` | uniformly between the bounds, both included; no log scale, no step | a bound that is not exactly a float (`10**400`, `2**53 + 1`), or a range that is not finite |
| `integer [low, high]` | every integer between the bounds | a bound beyond 2**53 in magnitude: the sampler draws integers through floats |
| `categorical` | one of the choices, as declared (string, boolean or number) | nothing |

A proposed candidate is checked against the parameter domains as any other candidate is. Its
values are written into the geometry as they are, with nothing rounded or snapped
(`9.797885082295046*mm`).

**Objectives.** The optimizer is told each objective's value, in the problem's order and with
its `direction`. One objective and several take the same path. The values are those of the
run's [assessment](evaluation.md#assessment-objectives-limits-and-feasibility): the evaluator's
measurement, as a float. A value that no float holds exactly (an integer beyond 2**53) stops
the study rather than being told as another value.

**Constraints.** Each bound of a constraint is a *limit*, named `stage/metric:bound`
(`simulation/ecal_barrel.angular_residual:max`). For each limit the optimizer is told its
physical margin (`value - max`, `min - value`, in the metric's unit), which is zero or below
when the limit is satisfied. `study.json` records this choice as `"constraint_values":
"physical margins"`. A limit is judged on the central value of the measurement, and its
uncertainty is recorded but does not decide. An infeasible candidate is a complete trial with
its real objective values, but it is left out of the Pareto front.

Optuna's TPE handles constraints by ranking infeasible trials by the sum of their positive
margins. While feasible trials are scarce (early in a study, or under tight limits), the least
infeasible trials join the "good" trials the sampler learns from. So constraints steer the
search toward feasibility, but they do not guarantee it.

- **Limits on one metric:** a value cannot be both below `min` and above `max`, so the sum is
  one margin, in that metric's unit, and the ranking means something.
- **Limits on several metrics:** the sum adds values in different units. `k4opt optimize` warns
  about this. Treat such a study as not yet scientifically mature: a characteristic `scale` per
  constraint is planned for it.

**The sampler.** Optuna's `TPESampler`, made explicitly with its seed and 10 startup trials,
both recorded, so that a change of Optuna's defaults cannot change a study. The first 10
trials of a study are random proposals, and TPE models the trials from then on.

## Trials and their outcomes

| The trial's record | Optuna trial | Values | Constraint values |
| --- | --- | --- | --- |
| `valid`, every limit satisfied | COMPLETE | the objectives | all zero or below |
| `valid`, a limit violated | COMPLETE | the objectives | at least one above zero |
| `invalid` | FAIL | none | none |
| `failed` | FAIL | none | none |
| interrupted, no record | FAIL | none | none |
| the trial stopped the study | FAIL | none | none |

An invalid or failed trial never gets made-up values (0, infinity) or constraint values. TPE
ignores FAIL trials, so a region that is always invalid can be proposed again. Treating
invalidity as a constraint would be a scientific choice, and it is not made.

Each Optuna trial carries `k4opt.*` user attributes:

- `k4opt.dir`, its directory;
- `k4opt.record`, the record it was told: its `id`, the SHA-256 of `summary.json` and its path
  relative to the study;
- `k4opt.status`, `k4opt.feasible`, `k4opt.message` (cut to 1000 characters; the record has
  all of it) and `k4opt.fingerprint` of its geometry;
- `k4opt.sampler_seed`, the seed of the session that proposed it;
- `k4opt.error` when it stopped the study, and `k4opt.recovered` when a resume finished it.

The Optuna study carries `k4opt.study` (the id of `study.json`), `k4opt.identity`,
`k4opt.parameters`, `k4opt.objectives` and `k4opt.constraints`.

**The study goes on** after an `invalid` or `failed` candidate, also after many in a row: a
failure can be a legitimate finding (ddsim refusing an overlapping geometry, say). Each trial
is printed as it ends, so a broken environment shows at once.

**The trial is FAIL with `k4opt.error`, and the study stops,** when:

- an evaluator's `check()` refuses the candidate, or anything else is wrong before its
  evaluation starts;
- a file the problem names changed since the study's session opened;
- a result breaks the evaluation contract (a `valid` result without a metric it owns, a wrong
  unit), or an evaluator raises anything but `EvaluationFailed` or `EvaluationInvalid`;
- the record does not hold up when it is read back (below);
- a value cannot be held by a float;
- writing fails, or Optuna or its storage fails (a database that goes away, an I/O error of the
  journal): this also stops a study outside a trial, as an `OptimizationError`, while a bug of
  k4opt propagates as itself.

Each evaluator's settings and collections are checked once, when the study opens and before
any file is created, so a mistake in them costs no trial. They need no candidate, so an
evaluator's own `check()` is only ever asked about candidates the sampler proposes.

**The optimizer is told what the record holds.** After each evaluation, `summary.json` is read
back from disk and checked:

- it was made for this study and trial;
- its candidate is the trial's, type and value;
- its evaluators, Key4hep stack, k4opt code and input files are the study's;
- it was made from the candidate alone (no supplied stage);
- its assessment holds up when recomputed from its raw measurements
  ([`Assessment.from_summary`](evaluation.md#assessment-objectives-limits-and-feasibility)).

Only then is Optuna told. This is also how a resume finishes a trial that an interrupted
session left with a record, so a recovered trial is told exactly what a live one would have
been.

## The study directory

```text
runs/cld-ecal-study/
├── study.json        what the study is made with, written once (k4opt.study/1)
├── study.journal     Optuna's storage, one JSON line per operation (unless --storage)
├── .lock             while a process works on the study: its host, process and start time
├── trial-000000/     what k4opt evaluate writes: geometry/, simulation/, evaluators/
│   └── summary.json  the record; "run": {"study": <study id>, "trial": 0}
└── trial-000001/ ...
```

Nothing in a trial directory is ever rewritten or reused, and nothing is deleted: every trial
keeps its geometry and its EDM4hep file.

`study.json` holds:

- the study's `id` and creation time;
- the problem's path and normalized definition;
- the path, size and SHA-256 of every file the problem names (`stages.*.file`, the files ddsim
  options name);
- each evaluator's name, version, source identity and declared metrics;
- the compact file, source root and fingerprint of the unmodified geometry;
- `KEY4HEP_STACK` and `K4GEO`;
- k4opt's version, the SHA-256 of its code and its git checkout;
- the `identity` digest;
- the optimizer: `optuna` and its version, the study name (`<problem name>-<8 hex>`), the
  storage (a URL without its password), the sampler and its seed, the directions, the
  parameters' distributions, the limits and the constraint-value policy.

## Resuming a study

`k4opt optimize ... --resume --output DIR --trials N` runs `N` more trials of the study in
`DIR`. A study resumes only with what it was made with, compared by content and never by
location. Moving the problem directory, an evaluator file, the k4geo tree or the study
directory itself is fine.

| Part | Compared: must be equal | Recorded only |
| --- | --- | --- |
| problem | its definition, but not its `name` | the path of the file |
| files the problem names | size and SHA-256 of each | their paths |
| evaluators | name, version, SHA-256 of the source file (one without a hash cannot make a study), the metrics it declares with their units (an instance may declare others) | path, whether the bytes were `executed`, the git revision (a change warns) |
| geometry | the fingerprint of the unmodified geometry, which covers every reachable file | compact path, source root, `K4GEO` |
| stack | `KEY4HEP_STACK` | |
| k4opt | the SHA-256 of its code | version, git commit |
| Optuna | its exact version; the study's directions, id, identity digest and limit names; the storage | |

The `identity` digest covers the compared column only. A resume is refused, with what changed,
for any difference in it. It is also refused for `--seed` with `--resume` (a study keeps its
seed), for a recorded optimizer (sampler, seed, startup trials, distributions, limits,
constraint-value policy) other than the one this k4opt would run the study with, and for a
directory that holds no `study.json`; using a study directory without
`--resume` is refused too. A changed Optuna version and changed k4opt code have no override:
install the recorded Optuna version, and run a long campaign from a k4opt checkout of its own.
A resume only warns when an evaluator's git checkout is at another commit or has other
uncommitted changes: the evaluator's own file is unchanged, but helper modules beside it may
not be. Helper modules outside a git checkout are not tracked at all.

Within a session, the files the problem names are checked before each candidate, the geometry
files by its materialization, and `KEY4HEP_STACK` and `K4GEO` before each candidate.

**Seeds.** Each session samples with a seed derived from the study's seed and the number of
trials already in the study. Otherwise a resumed session would replay the first session's
random proposals (with Optuna, the same seed repeats them). What this does not guarantee:

- TPE can still propose an earlier candidate, which is likely for integer and categorical
  parameters. It is evaluated again as a new trial with a record of its own, which spends
  budget but corrupts nothing, since the simulation seed repeats its result;
- an interrupted study does not follow the path an uninterrupted one would have taken.

Each trial records the seed of its session, and the same history, seed and Optuna version give
the same proposals again.

## Interruptions, the lock and recovery

**Ctrl-C and SIGTERM** (`systemctl --user stop`, a batch system ending a job) stop the study
cleanly. The running trial is FAIL, or is told its record if one was published just before;
ddsim is killed; the lock is removed; the command exits 1 and says how to resume.

**The lock.** `.lock` is created exclusively when a process starts working on a study, and
removed when it ends, so two processes never work on one study. A lock that a killed process
(SIGKILL, a crashed node) left behind is never taken over. The error names its host, process
and start time and, on the same host, whether that process still runs. Once nothing works on
the study any more, remove the file named and resume.

**Recovery.** Under the lock, before any new trial, a resume finishes the trials a session
left running:

- a trial without a directory, or whose directory has no `summary.json`, is FAIL; its
  directory is left as it is and never rerun or reused;
- a trial with a `summary.json` is told that record, after the same checks as a live one; a
  record that does not hold up makes the resume refuse.

Before that, every finished trial must still have the record the optimizer was told: the same
bytes as the SHA-256 in its `k4opt.record`. A record that is gone or was changed makes the
resume refuse, since nothing would show what the trial's values rest on. So does a trial
without any record, other than an interrupted one or one that stopped the study: another
client of the storage added it, and its values rest on no evaluation.

A `trial-*` directory that no trial of the study names makes the resume refuse. Recovered
trials do not count among the `--trials` of the resume, and interrupted trials are not retried
automatically.

## Storage

The default storage is `study.journal` in the study directory: Optuna's journal storage, one
JSON line per operation, appended and synced, with Optuna's own file lock. It needs no database
and no SQLAlchemy, and it is replayed when it is read, which is fine for thousands of trials.
`--storage URL` keeps the Optuna study in a database instead (SQLite on a local disk,
PostgreSQL), while the trial directories stay in the study directory. The database is reached,
and set up once, before the study directory is made, so a wrong URL creates nothing; a relative
SQLite path is relative to the working directory. Its password and the
values of its query parameters are never recorded or shown, also not in an error from the
database, and every resume needs the same URL.

## Python API

```python
from k4opt.evaluation import load_evaluator
from k4opt.optimization import optimize

report = optimize(
    "examples/problem/cld-ecal-granularity.yaml",
    [load_evaluator("examples/evaluators/calo_photon.py:CaloPhoton")],
    "runs/cld-ecal-study",
    trials=30,
    seed=42,  # or resume=True, storage="postgresql://..."
    warn=print,  # each warning
    progress=print,  # each TrialReport as its trial ends
)
report.pareto  # the trial numbers on the Pareto front of the feasible valid trials
report.trials  # TrialReport: number, directory, candidate, status, feasible, objectives,
# record (the summary's id), message, recovered
```

`optimize` raises `OptimizationError` (a `K4OptError`) when a study cannot start, resume or go
on, and `KeyboardInterrupt` after a clean stop on Ctrl-C or SIGTERM. Underneath it is an
[`EvaluationSession`](evaluation.md#evaluating-candidates-evaluationsession), which any other
optimizer can drive the same way.

## Not done

Retrying interrupted trials, reusing the record of a candidate proposed twice, a warm start from
`k4opt evaluate` records, sampler options, a characteristic `scale` per constraint, treating
invalid candidates as infeasible, parallel workers, a `study show` or export command, overrides
for changed software, and other optimizers. None of them is blocked by the design: records have
ids, studies reference them, and the session knows no optimizer.
