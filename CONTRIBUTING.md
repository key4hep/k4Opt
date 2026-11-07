# Contributing to k4opt

k4opt is a detector optimization framework for Key4hep. It is organized by
**component**: each component is a subpackage with its own public API, errors,
CLI command group, docs page and tests: `geometry`, `ddsim`, `k4run`, `software`, `evaluation`,
`problem`,
`record`, `execution`, `optimization`, `publication` and `visualization`, each depending only on
those before it (`problem` <- `record` <- `execution` <- `optimization` <- `publication`,
`visualization`). The Streamlit dashboard is no component: it lives in `dashboard/` and uses
`publication` and `visualization` like any user.

    k4opt/
      __init__.py      version + K4OptError, nothing else
      errors.py        error base classes shared by all components
      frozen.py        immutable copies of JSON-like data, shared by components that store it
      yamlio.py        strict YAML loading, shared by components that read YAML files
      jsonio.py        atomic publication of complete files: never over another, or replacing one whole
      hashing.py       files by content: each version of a file hashed once per process
      revision.py      git revisions and the identity of k4opt's own code, for provenance
      terminal.py      colors and widths of terminal output, shared by the command lines
      cli.py           the `k4opt` command; each component adds a command group
      <component>/     public API in __init__.py; errors.py, cli.py (register()), ...
    dashboard/         the Streamlit app, its container and pinned requirements
    openshift/         the dashboard's deployment on CERN OpenShift
    tests/
      unit/<component>/          no Key4hep stack needed
      integration/<component>/   need the stack ($K4GEO) and are skipped without it
      fixtures/<component>/      small realistic inputs
      support/                   helpers shared by tests
    docs/<component>.md          reference documentation of a component
    examples/<component>/        copy-pastable configs

Rules of thumb:

- Components share the top-level modules (`k4opt.errors`, `k4opt.frozen`,
  `k4opt.yamlio`, `k4opt.jsonio`, `k4opt.hashing`, `k4opt.revision`, `k4opt.terminal`). A component uses
  another through its public API (`k4opt.<component>`) or, inside k4opt, through that
  component's modules without a leading underscore (`k4opt.problem.plan`), never through its
  `_private` modules; imports go one way only (a unit test checks it).
- Users import from the component (`from k4opt.geometry import GeometrySource`),
  never from its private modules.
- The CLI is a thin wrapper over the library; it holds no logic.
- Keep the user-facing configuration as small as possible; ask whether a physicist
  describing an optimization problem needs a knob before adding one.
- Spelling is US English (enforced by codespell). Line length 100.
- `source setup.sh` sets up the Key4hep stack, a venv and the pre-commit hooks;
  `pre-commit run --all-files` must pass, as must `mypy` (strict), `pytest`.

## Before you push

```bash
source setup.sh                     # Key4hep stack, venv, editable install, git hooks
pre-commit run --all-files          # ruff, mypy (strict), codespell, shellcheck, actionlint, ...
python -m pytest -p xdist -n auto tests/unit   # every core; add tests/integration with $K4GEO
```

The visualization and dashboard tests need Plotly and Streamlit (and Optuna, to make the studies
they show), which the Key4hep stack lacks, and are skipped without them; CI runs them in its `dashboard` job, which
also type-checks `dashboard/` and builds and probes its container (see `docs/dashboard.md`):

```bash
pip install -r dashboard/requirements.txt && pip install --no-deps -e . && pip install optuna
python -m pytest -p xdist -n auto tests/unit/publication tests/unit/visualization tests/unit/dashboard
```
