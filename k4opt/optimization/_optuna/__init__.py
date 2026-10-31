"""Internal: Optuna behind :func:`k4opt.optimization.optimize` and
:func:`k4opt.optimization.load_study`, and the only package of k4opt that imports it.

``space`` is the problem in Optuna's terms (the search space, and the values a trial is told);
``storage`` is where the trials are kept (the journal or a database, opened for a run or read
only), and how its failures and secrets leave; ``optimizer`` is the optimizer ``study.json``
records and its Optuna study (made, resumed, checked); ``run`` is a study run by
:func:`optimize`; ``read`` is a study read back by :func:`load`, without changing it.
"""

from k4opt.optimization._optuna.read import load
from k4opt.optimization._optuna.run import optimize

__all__ = ["load", "optimize"]
