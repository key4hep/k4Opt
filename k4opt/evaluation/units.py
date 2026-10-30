"""The units a setting declared as a :class:`~k4opt.evaluation.Quantity` may be written in.

A value is converted exactly in decimal, by a power of ten, so ``70`` keV in the problem becomes
the same float as the literal ``70e-6`` in GeV. EDM4hep stores energies in GeV and lengths in
mm, so those are the units an evaluator usually declares.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

# the power of ten of each unit, per dimension
_UNITS: Mapping[str, Mapping[str, int]] = {
    "energy": {"eV": -9, "keV": -6, "MeV": -3, "GeV": 0, "TeV": 3},
    "length": {"um": -3, "mm": 0, "cm": 1, "m": 3},
}


def dimension(unit: str) -> str | None:
    """The dimension of ``unit`` (``energy``, ``length``), ``None`` for an unknown unit."""
    return next((name for name, units in _UNITS.items() if unit in units), None)


def units(name: str) -> tuple[str, ...]:
    """The units of the dimension ``name``."""
    return tuple(_UNITS[name])


def convert(value: int | float, unit: str, to: str) -> float:
    """``value`` in ``unit`` expressed in ``to``; both must be units of one dimension."""
    name = dimension(unit)
    assert name is not None
    assert dimension(to) == name
    return float(Decimal(repr(value)).scaleb(_UNITS[name][unit] - _UNITS[name][to]))
