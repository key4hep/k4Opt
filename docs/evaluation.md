# Evaluation contract

`k4opt.evaluation` defines how anything that measures a candidate (a simulation, a
reconstruction, an analysis, a cost model) talks to k4opt: **one protocol, one request, one
result**. This is the contract only: it does not run anything, look an evaluator up by the name in
`problem.yaml` or persist results. The evaluator that ships with k4opt and `k4opt evaluate` are
described in [evaluators.md](evaluators.md). An evaluator can be written and tested without Key4hep and without a problem file.

## The protocol

```python
from pathlib import Path
from k4opt.evaluation import EvaluationRequest, EvaluationResult, Evaluator, Measurement


class CellCount:
    name = "demo.cell_count"  # counterpart of evaluation.evaluator
    version = "1"  # counterpart of evaluation.version

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        size = request.candidate["constants.ECal_cell_size"]
        if not isinstance(size, float):
            return EvaluationResult("failed", message="ECal_cell_size must be a float")
        cells = 1000.0 / size**2
        return EvaluationResult(
            "valid", (Measurement("analysis", "ecal.cells", cells, unit="1", uncertainty=0.5),)
        )


evaluator: Evaluator = CellCount()
result = evaluator.evaluate(EvaluationRequest(candidate={"constants.ECal_cell_size": 5.0}))
```

`Evaluator` is a `typing.Protocol`: a class (or an object built by a function) with `name`,
`version` and `evaluate` is one. There is no base class, decorator or registry, and no
`SimulationEvaluator` or `AnalysisEvaluator`: those are scientific domains, not different
calling conventions. `name` and `version` identify the evaluator's measurement contract, as
`evaluation.evaluator` and `evaluation.version` do in the problem file; they are labels, not a
lookup mechanism. The *documentation* of an evaluator version defines what each of its
metrics means: selection, estimator, units, uncertainty convention.

## Request

`EvaluationRequest` is immutable and every field is optional:

| Field | Meaning |
| --- | --- |
| `candidate` | parameter values for this evaluation, keyed by the canonical parameter identity (a geometry parameter: its canonical target such as `constants.ECal_cell_size`); booleans, integers, finite floats or strings. Not checked against bounds |
| `settings` | the evaluator-defined `evaluation.settings`, nested data, copied and read-only |
| `inputs` | role name to local `Path` of an already prepared artifact (a compact XML snapshot, an EDM4hep file). The evaluator documents the roles it accepts |
| `workdir` | where the evaluator may write. Never created or checked by the request |
| `events`, `seed` | the optional controls of `evaluation`; `events` is positive |
| `backend_args` | tool name to raw arguments, kept exactly (order, repeats, empty strings), never parsed |

The request holds no `ProblemSpec`, geometry or backend object; the later runner builds it
from those.

## Result

`EvaluationResult` is what `evaluate` returns:

| Field | Meaning |
| --- | --- |
| `status` | `valid`: ran, result scientifically usable. `invalid`: ran, but the result is not usable (the evaluator's own checks failed); diagnostic measurements are allowed. `failed`: the execution failed |
| `measurements` | tuple of `Measurement`; may be empty and may mix stages |
| `artifacts` | role name to local output `Path`, possibly empty; a later evaluator can take them as `inputs`. Existence is not checked at construction |
| `message` | optional explanation, mainly for `invalid` and `failed` |
| `provenance` | optional JSON-compatible metadata the evaluator chose to supply |

A `Measurement` is `stage`, `metric`, a finite `value` (booleans, NaN and infinity are
rejected), an optional descriptive `unit` (never converted) and an optional symmetric,
nonnegative `uncertainty`. An omitted uncertainty means *not provided*, not zero. Each
`(stage, metric)` occurs once per result. A missing measurement is absent, never `0`, `inf` or
a fake success; a producer that only writes files returns a `valid` result with artifacts and
no measurements.

Violations raise `EvaluationContractError` (a `K4OptError`) when the object is built.

## Stage classifies, it does not prescribe

`stage` is a free descriptive string next to `metric`, as in the problem file. It is not a
workflow instruction, and the set of stages is not enumerated. One evaluator may report
`simulation` and `analysis` metrics together, or produce only artifacts.

## Problem, evaluator, record

- **Problem** (`k4opt.problem`): the declared parameter domains and the requested objective and
  constraint metrics.
- **Evaluator**: user-written code that accepts an `EvaluationRequest` and returns an
  `EvaluationResult`; the documentation of its version defines what its metrics mean.
- **`EvaluationRecord`**: the request and result of **one attempted evaluation**, plus what the
  framework observed.

| `EvaluationRecord` field | Source |
| --- | --- |
| `request`, `result` | the actual objects exchanged with the evaluator, unchanged |
| `evaluator_name`, `evaluator_version` | the evaluator object that ran, checked against the problem before the call |
| `candidate_key` | framework-computed, see below |
| `elapsed_seconds` | wall-clock time around the single `evaluate()` call (`time.perf_counter()`); not CPU time, memory or an estimate |

`result.provenance` is **evaluator-supplied**; the record's identity and elapsed time are
**framework-observed**. Neither proves which software environment ran: the problem's
`environment.key4hep` is a declared pin, and no executed command, input hash or version is
recorded yet.

`candidate_key` is a tuple with one `(reference, kind, tag, value)` entry per parameter, in the
problem's parameter order. It is problem-aware: `True`, `1` and `1.0` are not interchangeable
by Python equality here, so a categorical boolean, a categorical string `"1"` and a numeric `1`
stay distinct, while continuous `1`, `1.0` and `-0.0`/`0.0` (or numeric choices `1` and `1.0`)
agree. Numbers are written as exact decimal text, so huge integers are never forced through
`float`. The key is scoped to one problem definition; it is not a cross-problem hash, a
geometry fingerprint or a complete evaluation-cache key.

### Internal bridge (not the physicist-facing workflow)

`k4opt.problem._evaluate._evaluate_candidate(spec, candidate, evaluator, *, inputs=None,
workdir=None)` connects a `ProblemSpec` to an explicitly supplied evaluator. It is private,
meant for tests and a future runner; the user-facing entry point will take only the problem
YAML.

```python
record = _evaluate_candidate(spec, {"constants.ECal_cell_size": 6.0}, CellCount())
```

In order, it:

1. validates the candidate (`CandidateError` otherwise): exactly one value per declared
   parameter, keyed by `parameter.geometry.canonical_target`; continuous `int`/finite `float`
   within inclusive bounds; integer a real `int` (no `bool`, no rounding) within bounds;
   categorical one of the choices by kind (strings need a string, booleans a boolean, numbers
   compare numerically). Units are never interpreted or converted;
2. requires `evaluator.name` and `evaluator.version` to equal `evaluation.evaluator` and
   `evaluation.version` (`EvaluationContractError` otherwise), before anything runs;
3. builds the `EvaluationRequest` from the validated candidate plus `settings`, `events`,
   `seed` and the exact ordered `backend` arguments; `inputs` and `workdir` are passed through
   unchanged and nothing is created, inspected or materialized;
4. calls `evaluate()` exactly once. Exceptions of the evaluator propagate;
5. for a `valid` result, requires every `(stage, metric)` of any objective or constraint to be
   measured (extra metrics are fine) and, where a constraint declares a `unit`, the
   measurement's `unit` to be equal. A violation is an `EvaluationContractError`: an
   inconsistent evaluator, not a violated constraint. Whether a value satisfies `min`/`max` is
   not judged. `invalid` and `failed` results are kept as returned, even without metrics.

## Deliberately deferred

An evaluator registry or lookup by name, mapping `problem.yaml` to an implementation and
validating its metric names, `k4run`, a public execution API (`k4opt evaluate` is the one internal vertical slice),
chaining artifacts between evaluators (callers do it by hand, as the tests show), execution
cost, retries and scheduling, fingerprints and caching, persisting or serializing the `EvaluationRecord`,
and aggregation, significance or constraint judgment.
