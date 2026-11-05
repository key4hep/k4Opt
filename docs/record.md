# Record

`k4opt.record` is what the evaluation of a candidate found, as its run record `summary.json`
(schema `k4opt.run/1`) holds it: the result of each task of the candidate's plan, the
candidate's status and message derived from its required tasks, its measurements and an
assessment against the problem's objectives and constraints. A session writes it once all the
candidate's tasks ended ([execution](execution.md)); nothing else writes it, and it is never
changed afterwards.

```python
import json
from pathlib import Path

from k4opt.problem import load_problem
from k4opt.record import Assessment, peak_memory

summary = json.loads(Path("runs/7mm/summary.json").read_text())
assessment = Assessment.from_summary(load_problem("problem.yaml"), summary)
assessment.feasible  # True, False or None (not assessed)
peak_memory(summary)  # by task label, PeakMemory(process, program)
```

**A record is checked before anything trusts it.** `Assessment.from_summary(spec, summary)`
checks the record against its problem: its form, that it was made for this problem and
candidate (with the seeds of its `replicate`: 0, or the replicate of the candidate it was
evaluated on, see [execution](execution.md)), what each implementation declared and each binding gave it, each task's part against
the plan the problem gives (what it reads, whether and by what it is blocked, the files it made,
named relative to the run directory in its own directory), and the candidate's status, message,
required tasks, diagnostics and metrics recomputed from them; then it recomputes the assessment
from the measurements. What the implementations declared and returned is checked with the same
rules, and the same words, a live session uses (`k4opt.problem.contract`), so a record holds up
only where a live evaluation would have accepted it. It reads the record alone: neither the
evaluators nor any file the record names is needed, and none of `k4opt.execution` is imported. A
record that does not hold up raises `RecordError`, which says which part is at fault.

## Assessment: objectives, limits and feasibility

Each outcome carries an `Assessment` of the candidate, also written to the summary under
`assessment`:

| Evaluation | `objectives` | `limits` | `feasible` |
| --- | --- | --- | --- |
| `valid`, no constraints | the measurement of each objective, in the problem's order | none | `true` |
| `valid`, every limit satisfied | as above | each with `value`, `margin`, `scale`, `normalized`, `satisfied` | `true` |
| `valid`, a limit violated | as above | as above | `false` |
| `invalid` or `failed` | `null`, never a partial vector | each listed with its `scale`, `value`, `margin`, `normalized` and `satisfied` `null` | `null`: not assessed |
| a required supplied stage (`--input`) that a parameter reaches | as measured | as measured | `null` |

Each objective value (`ObjectiveValue`) and each limit names its metric: `evaluator`, `sample`
(`None` for an evaluator of the candidate), `stage`, `metric` and their `label`,
`sample/evaluator/stage/metric`. A limit is one bound of a constraint
(`gamma_10gev/calo.photon/simulation/ecal_barrel.angular_residual:max`), judged on
the central value of the measurement: `max` is satisfied when the value is at most the limit,
`min` when it is at least the limit, and equality satisfies both. The uncertainty is recorded
next to the value but does not decide. `margin` is `value - max` or `min - value` in the
metric's unit, zero or below when the limit is satisfied. `scale` is the constraint's
characteristic size of the metric ([problem](problem.md#constraints); 1 where it gives none),
and `normalized` the margin in units of it, `margin / scale`: dimensionless, what an optimizer
is told. Both are computed exactly and rounded once to the nearest float, so their sign is
always that of the comparison (a value nearer zero than any float is the smallest float of its
sign, never zero), and each is `null` if it does not fit in a float (the verdict stays exact). An infeasible candidate is a complete evaluation with real
objective values; it is only outside the limits.

`Assessment.from_summary(spec, summary)` recomputes the assessment of a `summary.json` as read
back: which tasks are required, from the problem; the candidate's status and message from its
required tasks', and its diagnostics from the others; each objective and constraint value from
the own measurements of the evaluation it names. The record's `implementations` must be the evaluators
the problem binds, with its versions and stages; its `samples` the problem's, each with the
evaluations of the bindings that measure it, and its `evaluations` those of the bindings that
measure the candidate, each of the evaluator the problem binds it to, with the `settings` (as `declared`) and `collections` the
problem gives it. What the candidate configures follows from the problem and the candidate
alone, so the record is checked against it: each stage's `definition` (the stage as the
candidate configures it, with the options its parameters set), each evaluation's
`settings.declared` (the problem's settings with what the parameters set), and the geometry's
`changes`, one per geometry target, in the problem's order (for a geometry that was made), each
naming its `parameter`,
the canonical `target` it names and the `new_value` the candidate gives it, each of
an attribute (`file`, `locator`, `attribute`) of its own. Where that attribute is and its
`old_value` need the geometry, which a record is read without: a session checks them as each
geometry is made, and a study against the attributes it resolved when it started. Each task
holds `varies`, the parameters whose values reach it (those that set it, and those that reach
what it reads), which must be those the problem gives. The geometry is `done`, with its
`compact`, `manifest`, `changes` and what its external references read, or `failed` with a
`message` and none of those, when its worker died or ran out of time before it was made: every
task that reads it is then `blocked` by it, and the candidate is `failed`. It checks again what a live evaluation enforces: the candidate is in the
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
trusting it. `represents_candidate` says only that no required stage a parameter reaches was
supplied; a supplied file is identified, not verified to be the stage the problem defines, so a
record with `supplied` other than `[]` is never fully determined by its problem and candidate.
