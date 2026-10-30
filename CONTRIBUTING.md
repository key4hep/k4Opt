# Contributing to k4opt

k4opt is a detector optimization framework for Key4hep. It is organized by
**component**: each component is a subpackage with its own public API, errors,
CLI command group, docs page and tests: `geometry`, `problem`, `evaluation`, `ddsim`
and `optimization`.

    k4opt/
      __init__.py      version + K4OptError, nothing else
      errors.py        error base classes shared by all components
      frozen.py        immutable copies of JSON-like data, shared by components that store it
      yamlio.py        strict YAML loading, shared by components that read YAML files
      jsonio.py        atomic publication of complete JSON records without overwriting
      revision.py      git revisions and the identity of k4opt's own code, for provenance
      terminal.py      colors and widths of terminal output, shared by the command lines
      cli.py           the `k4opt` command; each component adds a command group
      <component>/     public API in __init__.py; errors.py, cli.py (register()), ...
    tests/
      unit/<component>/          no Key4hep stack needed
      integration/<component>/   need the stack ($K4GEO) and are skipped without it
      fixtures/<component>/      small realistic inputs
      support/                   helpers shared by tests
    docs/<component>.md          reference documentation of a component
    examples/<component>/        copy-pastable configs

Rules of thumb:

- Components share only the top-level modules (`k4opt.errors`, `k4opt.frozen`,
  `k4opt.yamlio`, `k4opt.jsonio`, `k4opt.revision`, `k4opt.terminal`), never each other's
  internals; a component talks to another through its public API (`k4opt.<component>`).
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
python -m pytest tests/unit         # add tests/integration if $K4GEO is available
```
