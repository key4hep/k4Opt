#!/bin/bash
# Set up a Key4hep-based development environment for k4opt.
#
#   source setup.sh
#
# Mirrors key4hep/k4Bench: source the Key4hep stack (unless one is already
# active), create a CVMFS-aware virtual environment on top of it, install the
# development tools, install k4opt itself in editable mode and the pre-commit
# hooks. KEY4HEP_VERSION selects a stable release (or, failing that, a dated
# Spack nightly); unset, the latest nightly is used (as in k4Bench).

K4OPT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export K4OPT_REPO

if [ -z "$KEY4HEP_STACK" ]; then
    if [ -n "${KEY4HEP_VERSION:-}" ] && [ -d "/cvmfs/sw.hsf.org/key4hep/releases/${KEY4HEP_VERSION}" ]; then
        source /cvmfs/sw.hsf.org/key4hep/setup.sh -r "${KEY4HEP_VERSION}"
    elif [ -n "${KEY4HEP_VERSION:-}" ]; then
        source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack -r "${KEY4HEP_VERSION}"
    else
        source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
    fi
else
    echo "✅ KEY4HEP_STACK is already set. Skipping Key4HEP environment setup."
fi

# Python virtual environment layered on the CVMFS stack
mkdir -p ~/.local/bin
export PATH=~/.local/bin:"${PATH}"

if [ ! -f ~/.local/bin/cvmfs-venv ]; then
    curl -sL https://raw.githubusercontent.com/jbeirer/cvmfs-venv/main/cvmfs-venv.sh -o ~/.local/bin/cvmfs-venv
    chmod +x ~/.local/bin/cvmfs-venv
fi

if [ ! -d "${K4OPT_REPO}/py-venv" ]; then
    cvmfs-venv "${K4OPT_REPO}/py-venv" || return 1
else
    echo "✅ Virtual environment 'py-venv' already exists. Skipping creation."
fi

if [ "${VIRTUAL_ENV:-}" != "${K4OPT_REPO}/py-venv" ]; then
    echo "🔄 Activating virtual environment..."
    . "${K4OPT_REPO}/py-venv/bin/activate" || return 1
else
    echo "✅ Virtual environment is already active."
fi

# Development tools and k4opt itself (editable). Dependencies are never resolved
# by pip: they come from the Key4hep stack, and resolving them would make pip
# parse the metadata of every installed package (some are broken on CVMFS).
# --ignore-installed keeps pip from even looking at the stack's copies.
pip install --quiet --no-dependencies --ignore-installed -r "${K4OPT_REPO}/requirements.txt" || return 1
pip install --quiet --no-build-isolation --no-deps --ignore-installed -e "${K4OPT_REPO}" || return 1

# Some nightlies ship an unusable pytest package; fill the gap, again without
# dependency resolution.
if ! python -c "import pytest; pytest.__version__" &>/dev/null; then
    pip install --quiet --no-dependencies --ignore-installed \
        pytest pytest-cov coverage pluggy iniconfig packaging pygments
fi

# (Re)install the git hook on every run (not in CI, where nobody commits): the hook records the interpreter that
# installed it, so a hook created under a different stack or venv goes stale.
if [ -n "${CI:-}" ]; then
    echo "ℹ️  CI detected: skipping the git hook installation."
elif (cd "${K4OPT_REPO}" && python -m pre_commit install --install-hooks > /dev/null); then
    echo "✅ Pre-commit hooks installed."
else
    echo "⚠️  Could not install the pre-commit hooks; run 'python -m pre_commit install' manually."
fi

if [ -z "${K4GEO:-}" ] || [ ! -d "${K4GEO}" ]; then
    echo "⚠️  \$K4GEO='${K4GEO:-}' does not exist in this stack (incomplete nightly?)."
    echo "   Relative geometry paths will not resolve; try a dated nightly, e.g. KEY4HEP_VERSION=2026-10-20 source setup.sh"
fi
echo "✅ k4opt environment ready (K4GEO=${K4GEO:-unset})."
