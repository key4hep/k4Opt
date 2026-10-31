# Optimization: `k4opt optimize` and `k4opt study`

`k4opt optimize` runs a **study** of a problem. An optimizer proposes candidates. Each one is
evaluated exactly as [`k4opt evaluate`](evaluators.md#k4opt-evaluate) evaluates one, into a
trial directory of the study. The optimizer then learns each result from that trial's
`summary.json`, read back from disk and checked. The optimizer is
[Optuna](https://optuna.org)'s TPE sampler, but k4opt is not built around it. The problem file,
the evaluators and the run records know nothing of Optuna, and only one package of this
component (`k4opt/optimization/_optuna/`) imports it. A study keeps everything it was made with
in `study.json`, and it resumes only with exactly that. `k4opt study` reads a study back, to
inspect and export its results ([below](#inspecting-and-exporting-a-study)).

```bash
k4opt optimize examples/problem/cld-ecal-granularity.yaml \
  --evaluator examples/evaluators/calo_photon.py:CaloPhoton \
  --trials 30 --seed 42 --output runs/cld-ecal-study
```

It prints each trial as it ends, and the study at the end (the values here are illustrative):

```text
trial 0:   valid    feasible    constants.ECal_cell_size=9.79789  gamma_10gev/calo.photon/simulation/ecal_barrel.angular_residual=0.948  gamma_10gev/calo.photon/simulation/ecal_barrel.fired_cells=227
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
the time a trial takes and how the trials ended; on a narrow terminal the last of these go first.
A trial takes the mean time from the end of one trial to the end of the next, without the first
trial of the session, whose time also holds the study's setup. The time left is that of the
trials still to run, less the time the running one has had, so it counts down between the ends of
trials. When the study ends, the bar is left in place and marked `✔`, or `✘` when it was stopped;
a study that stops before any of its new trials ended (one that cannot start) leaves only its
error.
Piped or written to a file, the output is the plain lines above, as in a log; `NO_COLOR=1` turns
color off.

| Option | Meaning |
| --- | --- |
| `--evaluator FILE.py:Class` | an evaluator of the problem, as for `k4opt evaluate` (repeatable) |
| `--trials N` | how many *new* trials this invocation runs; each counts, whatever its outcome |
| `--output DIR` | the study directory: new or empty for a new study, the study's own with `--resume` |
| `--seed S` | the sampler's seed, in [0, 2**32); one is drawn and recorded without it. It is not the simulation's seed, which stays each sample's `simulation.seed` |
| `--resume` | continue the study in `DIR` (see [Resuming](#resuming-a-study)) |
| `--storage URL` | an Optuna database URL (SQLite on a local disk, PostgreSQL) in place of the journal file in `DIR`; it needs SQLAlchemy, and the same URL on every resume |
| `--timeout SECONDS` | kill `ddsim` if a trial's simulation runs longer than that: the candidate is `failed`, the trial FAIL, and the study goes on. No limit without it. It is recorded in each trial's record and is not part of the study, so a resume may give another one |

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
its `direction`. One objective and several take the same path. Each is named by its label,
`sample/evaluator/stage/metric`, with the sample and the binding that measures it
(`gamma_10gev/calo.photon/simulation/ecal_barrel.fired_cells`; `evaluator/stage/metric` for an
evaluator of the candidate; a `%` or `/` within a part is percent-escaped, so two objectives
never share a label). The values are those of the run's
[assessment](evaluation.md#assessment-objectives-limits-and-feasibility): that evaluation's
measurement, as a float. A value that no float holds exactly (an integer beyond 2**53) stops
the study rather than being told as another value.

**Constraints.** Each bound of a constraint is a *limit*, named by its metric's label and its
bound, `sample/evaluator/stage/metric:bound`
(`gamma_10gev/calo.photon/simulation/ecal_barrel.angular_residual:max`). For each limit the optimizer is told its
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
failure can be a legitimate finding (ddsim refusing an overlapping geometry, say). So does a
simulation killed at `--timeout`; without one, a candidate on which Geant4 stalls holds up an
unattended study until it is stopped. Each trial
is printed as it ends, so a broken environment shows at once.

**The trial is FAIL with `k4opt.error`, and the study stops,** when:

- an evaluator's `check()` refuses the candidate, or anything else is wrong before its
  evaluation starts;
- a file the problem names, a file the geometry's external references read, the environment or
  k4opt's code changed since the study's session opened, also while the trial ran;
- a result breaks the evaluation contract (a `valid` result without a metric an objective or a
  constraint names of it, a wrong unit), or an evaluator raises anything but `EvaluationFailed` or `EvaluationInvalid`;
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
- its evaluators, Key4hep stack, k4opt code, input files and the files the geometry's external
  references read are the study's;
- a file that only candidates name (a `ref` parameter choosing `${CAD}/other.stl`), which the
  study cannot know beforehand, reads as it did for every earlier trial that named it, in this
  session or an earlier one;
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
- k4opt's version, the SHA-256 of its code and its git checkout;
- the `identity` digest;
- the optimizer: `optuna` and its version, the study name (`<problem name>-<8 hex>`), the
  storage (a URL without its password), the sampler and its seed, the directions, the
  parameters' distributions, the limits and the constraint-value policy.

Combined objectives use `binding/stage/metric` without a sample prefix. Two combined bindings
reporting the same metric have separate objective columns. `study show` and `study trial`
include the role wiring. Changing sample properties, role wiring or declared role requirements
refuses resume, just as changing evaluator settings does. A failed or invalid upstream input
blocks the combined evaluation and produces a failed trial (the evaluation is incomplete), with no
partial objective vector.

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
| evaluators | each one the problem binds: name, version, SHA-256 of the source file (one without a hash cannot make a study), the metrics it declares with their units and the artifacts it declares (read from an instance, which may declare others than its class) | path, the reference it was loaded by, whether the bytes were `executed`, the git revision (a change warns) |
| geometry | the fingerprint of the unmodified geometry, which covers every reachable file below its root, and the SHA-256 of each file its external references read (`null` for one that names no file) | compact path, source root, the paths the external references resolve to, `K4GEO` |
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

Within a session, the files the problem names, the files the geometry's external references
read, `KEY4HEP_STACK`, `K4GEO` and k4opt's code are checked before each candidate and again
after it ran, before its record is written; the geometry's own files are checked by its
materialization. A change while a candidate runs leaves its directory without a
`summary.json`, and the trial stops the study.

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
ddsim is killed, with any process it started; the lock is removed; the command exits 1 and says
how to resume.

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
resume refuse, since nothing would show what the trial's values rest on. So does a trial that
does not hold what its record gives, checked as `load_study` checks it (its state, status,
candidate, objective values, feasibility and constraint values): the sampler would learn from
values no evaluation produced. So does a trial
without any record, other than an interrupted one or one that stopped the study, and a trial
without any mark of k4opt (each trial is marked with its directory when it is asked for, or as
`k4opt.recovered`): another client of the storage added it, and it would count among the
study's trials, and so change the next session's seed, without being one of its evaluations.
A trial that another client leaves RUNNING cannot be told from one of a killed k4opt process,
and recovery closes it as FAIL.

A study whose setup a killed process cut short, before its first trial, is set up anew or
finished by the resume. A `trial-*` directory that no trial of the study names makes the resume
refuse. Recovered trials are printed first, marked `(recovered)`, but they do not count among
the `--trials` of the resume, and interrupted trials are not retried automatically.

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

## Inspecting and exporting a study

`k4opt study` reads a study back as its records hold it. Nothing is evaluated, created, locked,
recovered or written, so it can read a study while a session works on it, a copy of the study
directory, or a study long after it ran: the problem file, the evaluators, the geometry and the
Key4hep stack it was made with need not exist any more. What the study optimizes comes from
`study.json`, which trials there are from the optimizer's storage, and every value from the
trials' `summary.json`.

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
directory:  /home/me/runs/cld-ecal-study (trials in study.journal)
parameters: constants.ECal_cell_size  continuous [2.5, 20.0] mm
objectives: gamma_10gev/calo.photon/simulation/ecal_barrel.angular_residual  minimize  (mrad)
            gamma_10gev/calo.photon/simulation/ecal_barrel.fired_cells  minimize  (1)
limits:     none
trials:     30: 28 valid (28 feasible, 0 infeasible), 1 invalid, 1 failed
pareto:     0, 4, 7
            trial 0:  constants.ECal_cell_size=9.79789 mm  gamma_10gev/calo.photon/simulation/ecal_barrel.angular_residual=0.948 mrad  gamma_10gev/calo.photon/simulation/ecal_barrel.fired_cells=227 1
            ...
```

`trial` prints one trial, chosen by its number, with every value as recorded: its status, its
candidate with units, each objective with its uncertainty and unit, each limit with the value it
judges, its bound, its margin and whether it is satisfied, the wall-clock time of the whole
evaluation and of each stage that ran, its record (path, id, SHA-256) and the message of an
invalid, failed or error trial. Every text from the study (names, units, paths, ids, messages)
is shown with its control characters escaped (`\x1b`), so neither a colored log in a message
nor anything else a study holds can change the terminal. Times are those the record
holds; nothing is estimated.

| The trial | Status | What is shown |
| --- | --- | --- |
| a valid record, every limit satisfied | `valid`, feasible | objectives, limits with margins |
| a valid record, a limit violated | `valid`, infeasible | the same; never on the front |
| an invalid or failed record | `invalid`, `failed` | no objective values; limits not assessed; its message |
| ended without a record | `interrupted` | nothing measured |
| stopped the study | `error` | nothing measured; why it stopped. A valid record whose values never reached the optimizer is named but not used |
| RUNNING in the optimizer | `running` | nothing measured or checked; a record already on disk is not read: the optimizer has not taken it yet, a resume will |

A running trial is being evaluated, or was left by a session that was killed; the resume closes
it. Counts cover every trial of the study, from all its sessions, recovered ones included.

**What is checked.** `study.json` must be `k4opt.study/1`, and its identity digest that of its
parts, so an edited problem definition is refused. The optimizer's study must be the one it
describes (id, identity, directions, limits). Every trial the optimizer was told a record of
must still have it, with the bytes whose SHA-256 it was told and the id it was told, made for
this trial of this study with its candidate, evaluators, Key4hep stack, k4opt code and files,
and its assessment must recompute from its raw measurements
([`Assessment.from_summary`](evaluation.md#assessment-objectives-limits-and-feasibility)).
What the optimizer holds must agree: a COMPLETE trial has a valid record with the same objective
values, feasibility and constraint values, a FAIL trial an invalid or failed one. A record that
is missing or does not hold up, a trial without any mark of k4opt, or a study that was never set
up makes the read fail, with the trial and file named: nothing is skipped, repaired or
recomputed with other code. A `trial-*` directory that no trial names is a warning. Warnings go
to stderr.

**The Pareto front** holds the feasible valid trials that no other one dominates: at least as
good in every objective, in its direction, and better in one. Trials with equal values all stay.
The recorded values are compared and their uncertainties play no part, so the front is the one
Optuna reports for the study.

**Storage.** The journal is read without Optuna's lock and never written. A study made with
`--storage URL` needs the URL again (`study.json` keeps it without its password); without it,
or with another, it is refused, and no journal is made in its place. The database itself refuses
every write: SQLite is opened in its read-only mode, and PostgreSQL over `psycopg2` or `psycopg`
in read-only transactions. A SQLite database in WAL mode is refused, since even a read-only
connection writes `-wal` and `-shm` files beside it there (`k4opt optimize` makes none in that
mode), and so is any other database, since nothing would keep it from being written. A database that Optuna would set up or repair as it opens it (its tables, version row
or schema revision missing) is refused rather than repaired. The URL's secrets are never shown.

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
| `status` | `valid`, `invalid`, `failed`, `interrupted`, `error` or `running` |
| `feasible` | `true` or `false` for a valid trial, empty otherwise |
| `pareto`, `recovered` | `true` for a trial on the front, for one a resume finished |
| `parameter:<reference>` | the candidate's value (its unit is the problem's) |
| `objective:<sample>/<evaluator>/<stage>/<metric>`, `…:unit`, `…:uncertainty` | the measured value, its unit, its uncertainty (without `<sample>/` for an evaluator of the candidate) |
| `limit:<sample>/<evaluator>/<stage>/<metric>:<bound>:value`, `…:margin`, `…:satisfied` | the value judged, its margin (zero or below when satisfied), the verdict |
| `elapsed_seconds`, `stage:<sample>/<stage>:elapsed_seconds` | the recorded wall-clock time of the evaluation, and of each stage of each sample that ran (`stage:gamma_10gev/simulation:elapsed_seconds`) |
| `record`, `record_sha256`, `record_path` | the record's id, SHA-256 and path in the study |
| `message` | why the trial is invalid, failed or an error |

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

**JSON** (`k4opt.study-results/1`): the whole study, typed, with `study.json` as it is.

```json
{
  "format": "k4opt.study-results/1",
  "directory": "/home/me/runs/cld-ecal-study",
  "study": {"schema": "k4opt.study/1", "id": "3f9c2a1b-...", "problem": {}, "optimizer": {}},
  "outcomes": {"feasible": 28, "infeasible": 0, "invalid": 1, "failed": 1, "interrupted": 0, "error": 0, "running": 0},
  "pareto": [0, 4, 7],
  "warnings": [],
  "trials": [
    {
      "number": 0,
      "status": "valid",
      "outcome": "feasible",
      "recovered": false,
      "candidate": {"constants.ECal_cell_size": 9.797885082295046},
      "assessment": {"feasible": true, "objectives": [], "limits": []},
      "elapsed_seconds": 291.3,
      "stage_seconds": {"gamma_10gev/generation": 0.4, "gamma_10gev/simulation": 280.2},
      "message": null,
      "record": {"id": "7a2aca36-...", "path": "trial-000000/summary.json", "sha256": "a22d51..."},
      "pareto": true
    }
  ]
}
```

A trial's `assessment` is the one its `summary.json` holds (`null` without a record): each
objective as `stage`, `metric`, `value`, `unit` and `uncertainty`, and each limit as `stage`,
`metric`, `bound`, `limit`, `unit`, `value`, `uncertainty`, `margin` and `satisfied`.

## Python API

```python
from k4opt.evaluation import load_evaluator
from k4opt.optimization import optimize

report = optimize(
    "examples/problem/cld-ecal-granularity.yaml",
    [load_evaluator("examples/evaluators/calo_photon.py:CaloPhoton")],  # the class
    "runs/cld-ecal-study",
    trials=30,
    seed=42,  # or resume=True, storage="postgresql://..."
    warn=print,  # each warning
    progress=print,  # each TrialReport as its trial ends, the recovered ones of a resume first
)
report.pareto  # the trial numbers on the Pareto front of the feasible valid trials
report.trials  # TrialReport: number, directory, candidate, status, feasible, objectives,
# record (the summary's id), message, recovered
```

`optimize` raises `OptimizationError` (a `K4OptError`) when a study cannot start, resume or go
on, and `KeyboardInterrupt` after a clean stop on Ctrl-C or SIGTERM. Underneath it is an
[`EvaluationSession`](evaluation.md#evaluating-candidates-evaluationsession), which any other
optimizer can drive the same way.

`load_study` reads a study back, as `k4opt study` does:

```python
from k4opt.optimization import load_study

study = load_study("runs/cld-ecal-study")  # storage="postgresql://..." for a database study
study.name, study.id, study.created, study.directory
study.problem  # the ProblemSpec of study.json: parameters (bounds or choices, unit),
# objectives (direction), constraints
study.outcomes  # how many trials ended how: feasible, infeasible, invalid, failed, ...
study.pareto  # the numbers of the feasible valid trials on the Pareto front

trial = study.trial(12)  # by its number
trial.status, trial.outcome, trial.candidate, trial.message, trial.recovered
for m in trial.assessment.objectives:  # None for a trial without a valid result
    print(m.stage, m.metric, m.value, m.unit, m.uncertainty)
trial.assessment.limits  # each Limit: value, limit, margin, satisfied
trial.elapsed_seconds, trial.stage_seconds  # recorded times, stages by label
trial.record, trial.sha256, trial.summary_path  # the summary.json it was read from

data = study.to_dict()  # the JSON export, plain data
```

`StudyResults` and `TrialResult` are a snapshot: immutable, detached from the storage, and free
of Optuna types; `study.document` is `study.json` as a read-only mapping. They are not the
`StudyReport` and `TrialReport` of `optimize`, which report a study as it runs. `load_study`
raises `OptimizationError` for anything it cannot read or does not trust (above), and needs
Optuna, which holds the trials; the rest of k4opt does not.

## Not done

Retrying interrupted trials, reusing the record of a candidate proposed twice, a warm start from
`k4opt evaluate` records, sampler options, a characteristic `scale` per constraint, treating
invalid candidates as infeasible, parallel workers, overrides for changed software, other
optimizers, comparing or plotting studies, and exporting an evaluator's other measurements (each
row names its record, which holds them). None of them is blocked by the design: records have
ids, studies reference them, and the session knows no optimizer.
