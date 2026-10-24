# Evaluation contract

`k4opt.evaluation` defines how anything that measures a candidate (a simulation, a
reconstruction, an analysis, a cost model) talks to k4opt: **one protocol, one request, one
result**. This is the contract only. Nothing runs `ddsim` or `k4run`, looks an evaluator up
by the name in `problem.yaml`, or records results; `load_problem()` still only resolves the
geometry. An evaluator can be written and tested without Key4hep and without a problem file.

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

## Evaluator-supplied versus framework-captured

`EvaluationResult` carries only what the evaluator reports. Software versions, hashes of
inputs, executed commands, timings and the candidate identity are not collected
automatically; a future **`EvaluationRecord`** will add what k4opt itself observes. The two
are different objects and not interchangeable. The record is not implemented.

## Deliberately deferred

An evaluator registry or lookup by name, mapping `problem.yaml` to an implementation and
validating its metric names, running `ddsim`/`k4run`, geometry snapshots for a candidate,
chaining artifacts between evaluators (callers do it by hand, as the tests show), execution
cost, retries and scheduling, fingerprints and caching, the persisted `EvaluationRecord`,
and aggregation, significance or constraint judgment.
