# k4opt

A detector optimization framework for [Key4hep](https://key4hep.github.io/key4hep-doc/).

k4opt is being built to let detector experts define, evaluate, compare and share
detector optimization problems for the Key4hep chain (generation, simulation,
reconstruction, analysis), starting from the detector descriptions in
[k4geo](https://github.com/key4hep/k4geo). It is in early development: **geometry handling and a first end-to-end evaluation
(one candidate, simulated with `ddsim`) exist; there is no optimizer yet.**

## Components

k4opt is organized by component. Each lives in its own subpackage with its own
API, command group, documentation and tests.

| Component | Purpose | Status |
| --- | --- | --- |
| [`k4opt.geometry`](docs/geometry.md) | Discover addressable parameters of a DD4hep XML geometry, override them, materialize modified geometries | available |
| problem definition | Declare which parameters vary and within which bounds, and what is evaluated. A [draft specification](docs/problem.md) can be loaded, validated and its geometry targets resolved (`k4opt.problem.load_problem`, against the active `$K4GEO`); a single candidate can be evaluated (`k4opt evaluate`); a general evaluator registry and further metrics are **proposed, not yet implemented** | in progress |
| [`k4opt.evaluation`](docs/evaluation.md) | The contract an evaluator implements (request, result, measurements); evaluators only read the output of a pipeline stage that the framework runs. `load_evaluator` loads an evaluator from a file; a per-run `summary.json` exists; a general runner and result store are **not implemented** | in progress |
| [`k4opt.ddsim`](docs/ddsim.md) | Run `ddsim` on a prepared geometry with a particle gun or an input sample: command from the problem's `generation` and `backend`, protected options, logs; run once per candidate as the `simulation` stage | first vertical slice |
| [examples](docs/evaluators.md) | One problem and one evaluator to copy from, run with `k4opt evaluate PROBLEM --evaluator FILE.py:Class --param REF=VALUE --output DIR`. The problem asks how fine the CLD ECAL cells should be: finer cells measure a photon's direction better (`calo.photon` measures it) but fire more readout channels. k4opt ships no evaluators | example |
| optimization | Drive the evaluation with existing optimization libraries | planned |

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

## A first look: the geometry component

```yaml
# examples/geometry/cld.yaml
geometry: FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml   # relative to $K4GEO

overrides:
  - target: constants.InnerTracker_Barrel_radius_1
    value: "405*mm"
  - target: InnerTrackerBarrel.layer[id=1].rphi_layout.nphi
    value: "90"
```

```bash
k4opt geometry inspect examples/geometry/cld.yaml --filter InnerTracker_Barrel_radius
k4opt geometry apply examples/geometry/cld.yaml --output ./generated/cld-test --dry-run
k4opt geometry apply examples/geometry/cld.yaml --output ./generated/cld-test
ddsim --compactFile ./generated/cld-test/FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml ...
```

The values are technical examples only, not validated detector variants. The full
guide (target references, include handling, override semantics, manifest, Python
API, limitations) is in [docs/geometry.md](docs/geometry.md).

## Development

```bash
source setup.sh                       # Key4hep stack + venv + editable install + git hooks
python -m pytest tests/unit           # no stack needed
python -m pytest tests/integration    # needs $K4GEO (or K4GEO_DIR=/path/to/k4geo)
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
