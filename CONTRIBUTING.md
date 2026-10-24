# Contributing to k4opt

k4opt is a detector optimization framework for Key4hep. It is organized by
**component**: each component is a subpackage with its own public API, errors,
CLI command group, docs page and tests. `geometry` is the first one; the
problem definition, evaluation and optimization components will sit beside it.

    k4opt/
      __init__.py      version + K4OptError, nothing else
      errors.py        error base classes shared by all components
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

- Components depend on `k4opt.errors` only, never on each other's internals; a
  component talks to another through its public API (`k4opt.<component>`).
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
