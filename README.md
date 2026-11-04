# k4opt

A detector optimization framework for [Key4hep](https://key4hep.github.io/key4hep-doc/).

👉 [Open the live dashboard](https://k4opt-dashboard.app.cern.ch/)

k4opt is being built to let detector experts define, evaluate, compare and share
detector optimization problems for the Key4hep chain (generation, simulation,
reconstruction, analysis), starting from the detector descriptions in
[k4geo](https://github.com/key4hep/k4geo). It is in early development: **geometry handling, a first end-to-end evaluation (a
candidate, simulated with `ddsim` once and measured by one or more evaluators), a first
optimization with Optuna, and publishing its results for a first dashboard exist.**

## Components

k4opt is organized by component. Each lives in its own subpackage with its own
API, command group, documentation and tests.

| Component | Purpose | Status |
| --- | --- | --- |
| [`k4opt.geometry`](docs/geometry.md) | Discover addressable parameters of a DD4hep XML geometry, override them, materialize modified geometries | available |
| [`k4opt.problem`](docs/problem.md) | Declare which parameters vary and within which bounds, what each candidate is measured on (named `samples`, each with its stages, each stage run by an executor or given as a file) and who measures it (`evaluators`). A draft specification can be loaded, validated and its geometry targets resolved (`k4opt.problem.load_problem`, against the active `$K4GEO`); a general evaluator registry and further metrics are **proposed, not yet implemented** | in progress |
| [`k4opt.evaluation`](docs/evaluation.md) | The contract an evaluator implements (request, result, measurements, what it declares); evaluators read a pipeline stage or combine per-sample measurements and artifacts through declared roles. `load_evaluator` loads an evaluator from a file | in progress |
| [`k4opt.execution`](docs/execution.md) | Evaluate candidates end to end: `EvaluationSession` (and `k4opt evaluate` for one candidate) materializes each candidate's geometry, runs each sample's stages once and calls every evaluator, inline or in worker processes, and publishes the candidate's record | in progress |
| [`k4opt.record`](docs/record.md) | What a candidate's evaluation found, as its `summary.json` holds it: task results, status, measurements and the assessment against the objectives and constraints, read back and checked without the evaluators (`Assessment.from_summary`) | in progress |
| [`k4opt.ddsim`](docs/ddsim.md) | Run `ddsim` on a prepared geometry with a particle gun or an event file: command from a sample's stages (the ddsim options written the way ddsim reads them), protected options, logs; run once per candidate and sample as its `simulation` stage, whatever number of evaluators read it | first vertical slice |
| [`k4opt.k4run`](docs/k4run.md) | Run a Gaudi reconstruction (`k4run`, such as CLD's) from a copy of its directory on the candidate's geometry and every simulated event: the stage's options as the job's arguments, checked by the job's own dry run; the geometry, input and output it read verified from its resolved options; run once per candidate and sample as its `reconstruction` stage | reconstruction stage |
| [examples](docs/evaluators.md) | Runnable photon-direction, photon/pi0-separation and reconstructed-pi0 (particle flow) problems with evaluators to copy from, run with `k4opt evaluate PROBLEM --evaluator FILE.py:Class [--evaluator ...] --param NAME=VALUE --output DIR`. The problem asks how fine the CLD ECAL cells should be: finer cells measure a photon's direction better (`calo.photon` measures it) but fire more readout channels. k4opt ships no evaluators | example |
| [`k4opt.optimization`](docs/optimization.md) | Search a problem's parameters with an existing optimizer (Optuna's TPE sampler): `k4opt optimize` proposes candidates, evaluates each into a trial directory, judges them against the objectives and constraints, and resumes a study only with what it was made with; `k4opt study` (`load_study`) reads a study back, checked and unchanged, to show its trials and Pareto front or export them as CSV or JSON | first version |
| [`k4opt.publication`](docs/publication.md) | Publish a study as a portable, read-only snapshot without any local path (`k4opt publish STUDY --output DIR`) into a directory that can be served as static files or copied anywhere, with a catalog of its studies, and read it back from a directory or an HTTP(S) URL, checked, without Optuna | first version |
| [`k4opt.visualization`](docs/visualization.md) | Plotly figures of a study from its results alone: `trials_figure`, the trials on two objectives or parameters as feasible, infeasible or on the Pareto front, with units and uncertainties (`k4opt[plot]`) | first version |
| [dashboard](docs/dashboard.md) | A Streamlit app in `dashboard/` that shows the studies of a publication, with its container, CERN OpenShift manifests and a gated deploy workflow; deployed as the [live dashboard](https://k4opt-dashboard.app.cern.ch/) | first version |

The planned components are directions, not commitments; their shape depends on
what the first ones teach.

Design principles:

- as easy as possible for the physicist running experiments: small configuration files
- detector agnostic: CLD is the prototype, nothing in the core is CLD-specific
- reuse existing open-source tools instead of rebuilding them
- every experiment reproducible and identifiable
- stay independent of the executor (local, batch, grid)

## Installation

k4opt runs inside the Key4hep stack, which provides Python, lxml, PyYAML and
`$K4GEO` (the installed k4geo tree). The setup mirrors
[k4Bench](https://github.com/key4hep/k4Bench):

```bash
git clone https://github.com/key4hep/k4Opt.git && cd k4Opt
source setup.sh      # sources Key4hep (latest nightly unless KEY4HEP_VERSION is set),
                     # creates a CVMFS-aware venv, installs k4opt (editable) and the pre-commit hooks
k4opt --help
```

`setup.sh` is idempotent; source it in every new shell. If the default nightly is
incomplete (the script warns when `$K4GEO` is missing) or you need a fixed stack,
pin one: `KEY4HEP_VERSION=2026-10-20 source setup.sh` (a dated Spack nightly) or
`KEY4HEP_VERSION=2026-04-08 source setup.sh` (a release).

Without CVMFS you can still develop and run the unit tests in any Python ≥ 3.13
environment with `pip install -e ".[dev]"`.

## A first look

Find the parameter to vary in a geometry from k4geo:

```bash
k4opt geometry inspect FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml --filter ECal_cell_size
```

```text
FCCee/CLD/compact/CLD_o2_v09/ECalBarrel_o2_v01_03.xml  (4)
     ECal_cell_size.name                            ECal_cell_size
     constants.ECal_cell_size                       5.1*mm
     ...
```

name it in a problem, here
[examples/problem/cld-ecal-granularity.yaml](examples/problem/cld-ecal-granularity.yaml):

```yaml
geometry: FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml   # relative to $K4GEO

parameters:
  ecal_cell_size:            # the name candidates and records use
    type: continuous
    bounds: [2.5, 20.0]
    unit: mm
    targets:                 # what it varies: here, a constant of the geometry
      geometry: ECal_cell_size
```

say what each candidate is measured on, and who measures it:

```yaml
samples:
  gamma_10gev:
    generation:
      executor: gun          # 10 GeV photons into the barrel (see the example for the keys)
      particle: gamma
      # ...
    simulation:
      executor: ddsim
      events: 1000
      seed: 42
      physics.list: FTFP_BERT

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
```

and evaluate one candidate (about 5 minutes on one core):

```bash
k4opt evaluate examples/problem/cld-ecal-granularity.yaml \
    --evaluator examples/evaluators/calo_photon.py:CaloPhoton \
    --param ecal_cell_size=10.0 --output runs/cld-ecal-10mm
```

(`--dry-run` instead of `--output` shows what the candidate sets, and simulates nothing). A
parameter can also target a stage option or an evaluator setting, or several of these:
[cld-ecal-bindings.yaml](examples/problem/cld-ecal-bindings.yaml) varies the cell size, a ddsim
option and the evaluator's threshold together.

or let an optimizer search the cell size, one evaluation per trial:

```bash
k4opt optimize examples/problem/cld-ecal-granularity.yaml \
    --evaluator examples/evaluators/calo_photon.py:CaloPhoton \
    --trials 30 --seed 42 --output runs/cld-ecal-study
```

and look at the results later, without running anything again:

```bash
k4opt study show runs/cld-ecal-study      # what it optimizes, its trials and Pareto front
k4opt study trial runs/cld-ecal-study 12  # one trial, every value as recorded
k4opt study export runs/cld-ecal-study --format csv --output results.csv
```

publish it, and browse it in the dashboard:

```bash
k4opt publish runs/cld-ecal-study --output published/
K4OPT_DASHBOARD_DATA=published/ streamlit run dashboard/app.py   # pip install -r dashboard/requirements.txt
```

The guides: [geometry](docs/geometry.md) (target references, include handling, override
semantics, manifest, Python API, limitations), [problem](docs/problem.md),
[evaluation](docs/evaluation.md), [execution](docs/execution.md), [record](docs/record.md),
[example evaluators](docs/evaluators.md), [optimization](docs/optimization.md),
[publication](docs/publication.md), [visualization](docs/visualization.md) and
[dashboard](docs/dashboard.md). The [photon/pi0 example](docs/evaluators.md#photonpi0-example)
combines two samples through histogram artifacts and weighs the pi0 rejection of finer cells
against an approximate channel count.

## Development

```bash
source setup.sh                       # Key4hep stack + venv + editable install + git hooks
python -m pytest -p xdist -n auto tests/unit          # every core; no stack needed
python -m pytest -p xdist -n auto tests/integration   # needs $K4GEO (or K4GEO_DIR=/path/to/k4geo)
pre-commit run --all-files            # ruff, mypy (strict), codespell, shellcheck, actionlint, ...
```

The pre-commit hooks run on every `git commit` once `setup.sh` has installed
them, and CI runs the same hooks plus the tests inside the Key4hep container
(`.github/workflows/ci.yml`). Dependency updates are proposed weekly by
dependabot.

Repository layout and conventions for adding a component are described in
[CONTRIBUTING.md](CONTRIBUTING.md): `k4opt/<component>/` for code, `tests/{unit,integration}/<component>/`,
`docs/<component>.md` and `examples/<component>/`.

## License

MIT, see [LICENSE](LICENSE).
